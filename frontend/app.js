const $ = (id) => document.getElementById(id);
const sourceNames = {rule_nlu:'本地规则', semantic_nlu:'本地语义', nlu_guard:'边界检查', langchain_agent:'云端 Agent', offline_script:'离线处理'};
const intentNames = {set_temperature:'设置空调温度', query_vehicle:'查询车辆状态'};
const reasons = {
  exact_rule_match:'完整命中规则，未调用向量模型。', semantic_vehicle_query:'语义匹配通过，查询当前模拟车辆状态。',
  validated_absolute_temperature:'明确的绝对温度请求，已通过写操作检查。', below_similarity_or_margin_threshold:'相似度或意图分差不足，本地 NLU 未接受。',
  missing_temperature:'缺少明确的目标温度，需要进一步澄清。', relative_temperature_unsupported:'当前不支持相对调节，请提供绝对温度。',
  negated_or_hypothetical:'检测到否定或假设表达，不授予本地执行权限。', question_or_consultation:'疑问或咨询表达，不作为确定车控指令。',
  quoted_or_reported:'引述或讨论内容，不直接执行。', compound_or_trip:'多步任务或出行请求，交给后续处理。',
  deferred_action:'当前不支持延后或定时执行。', unsupported_domain:'不在当前本地 NLU 支持范围内。',
  compound_or_observation:'包含其他操作或状态描述，不直接修改温度。', read_request_not_write:'查询表达不授予温度写权限。',
  multiple_or_ambiguous_numbers:'存在多个或不明确的目标数值。', not_an_explicit_temperature_command:'缺少明确的温度设置表达。',
  no_rule_match:'未命中现有规则。', input_too_long:'输入超过本地 NLU 长度范围。',
};
let state = null;
let busy = false;
let toastTimer;

function node(tag, className = '', text = '') {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = text;
  return element;
}

function spark() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'icon');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#i-spark');
  svg.append(use);
  return svg;
}

function toast(message) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').hidden = false;
  toastTimer = setTimeout(() => {$('toast').hidden = true;}, 6000);
}

async function request(path, data) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await fetch(path, {
      method: data === undefined ? 'GET' : 'POST',
      headers: data === undefined ? {} : {'Content-Type': 'application/json'},
      credentials:'same-origin', signal:controller.signal,
      body: data === undefined ? undefined : JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '请求未完成，请稍后重试。');
    return result;
  } catch (error) {
    if (error.name === 'AbortError' || error instanceof TypeError) {
      throw new Error('连接中断或响应超时。操作可能仍在处理，请先刷新状态，避免重复提交。');
    }
    throw error;
  } finally {clearTimeout(timer);}
}

function setBusy(value, chat = false) {
  busy = value;
  const disabled = busy || !state;
  for (const id of ['mode', 'nlu-mode', 'reset', 'prompt']) $(id).disabled = disabled;
  // 首次连接失败也允许手动重试，不能因尚无会话而锁住刷新按钮。
  $('refresh').disabled = busy;
  document.querySelectorAll('[data-prompt]').forEach(button => {button.disabled = disabled;});
  $('send').disabled = disabled || !$('prompt').value.trim();
  $('pending').hidden = !(busy && chat);
  $('pending-label').textContent = state?.mode === 'live' ? '正在识别指令并处理任务…' : '正在本地识别并执行…';
}

function duration(ms) {return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(1)} ms`;}

function messageElement(message) {
  const user = message.role === 'user';
  const article = node('article', `message ${user ? 'user' : 'assistant'}${message.error ? ' error' : message.rejected ? ' rejected' : ''}`);
  const avatar = node('span', 'avatar', user ? '我' : '');
  if (!user) avatar.append(spark());
  const body = node('div', 'message-body');
  const header = node('div', 'message-header');
  header.append(node('span', '', user ? '你' : '座舱助手'));
  if (!user) header.append(node('span', 'message-source', message.error ? '调用失败' : message.rejected ? '操作已拒绝' : sourceNames[message.route] || '需要澄清'));
  header.append(node('span', 'message-time', message.at || ''));
  body.append(header, node('div', 'message-content', message.content));
  article.append(avatar, body);
  return article;
}

function eventSummary(event) {
  const d = event.data;
  switch (event.event) {
    case 'nlu.result': return ['NLU 完成识别', `${intentNames[d.intent] || '未确定意图'} · ${d.action === 'execute' ? '本地接受' : d.action === 'clarify' ? '需要澄清' : '本地未接受'}`, ''];
    case 'route': return [`进入${sourceNames[d.path] || d.path}`, d.note || '', ''];
    case 'agent.tools_selected': return ['模型选择工具', d.calls.map(c => c.name).join('、'), ''];
    case 'tool.call': return [`调用 ${d.name}`, Object.keys(d.arguments).length ? JSON.stringify(d.arguments) : '无需参数', ''];
    case 'tool.result': {
      if (!d.result.ok) return ['操作被拦截', d.result.error.message, 'error'];
      const data = d.result.data;
      const summary = data.routes ? `返回 ${data.routes.length} 条路线` : data.chargers ? `返回 ${data.chargers.length} 个充电站` : Object.entries(data).map(([k,v]) => `${k}: ${v}`).join(' · ');
      return ['工具执行成功', summary, 'success'];
    }
    case 'agent.error': return ['模型调用失败', d.message, 'error'];
    default: return [event.event, '', ''];
  }
}

function renderAnalysis(last) {
  const d = last?.nlu;
  $('nlu-empty').hidden = !!d;
  $('nlu-detail').hidden = !d;
  const badge = $('decision-badge');
  badge.className = 'mini-badge';
  badge.textContent = !d ? '等待输入' : last.error ? '模型出错' : last.rejected ? '参数已拦截' : d.action === 'execute' ? '本地接受' : d.action === 'clarify' ? '需要澄清' : '交给后续处理';
  if (d) {
    badge.classList.add(last.error ? 'error' : last.rejected || d.action !== 'execute' ? 'warning' : 'accepted');
    $('result-source').textContent = sourceNames[d.source] || d.source;
    $('result-intent').textContent = intentNames[d.intent] || '—';
    $('result-arguments').textContent = Object.keys(d.arguments || {}).length ? JSON.stringify(d.arguments) : '无';
    $('similarity').textContent = d.similarity == null ? '—' : d.similarity.toFixed(4);
    $('margin').textContent = d.margin == null ? '—' : d.margin.toFixed(4);
    $('nlu-time').textContent = duration(d.elapsed_ms || 0);
    $('match-box').hidden = !d.matched_example;
    $('matched-example').textContent = d.matched_example || '';
    $('decision-reason').textContent = reasons[d.reason] || d.reason || '';
  }
  const events = last?.events || [];
  $('trace-empty').hidden = events.length > 0;
  $('timeline').replaceChildren(...events.map(event => {
    const [title, detail, type] = eventSummary(event);
    const li = node('li', type);
    li.append(node('strong', '', title), node('p', '', detail));
    return li;
  }));
  $('total-time').textContent = last ? duration(last.duration_ms) : '—';
  $('raw-details').hidden = !last;
  $('raw-events').textContent = JSON.stringify(events, null, 2);
}

function render(next, scroll = false) {
  state = next;
  $('mode').value = next.mode;
  $('nlu-mode').value = next.nlu;
  $('mode-badge').textContent = next.mode === 'offline' ? '离线演示' : '真实模型模式';
  $('nlu-badge').textContent = next.nlu === 'hybrid' ? 'BGE 语义 + 规则' : '本地规则';
  $('mode-note').textContent = next.mode === 'offline' ? '本地处理文字指令；机场出行使用固定脚本演示。' : '本地未接受的请求会交给云端 Agent，并携带最近对话上下文。';
  $('battery').replaceChildren(document.createTextNode(String(next.vehicle.battery_percent)), node('small', '', '%'));
  $('temperature').replaceChildren(document.createTextNode(String(next.vehicle.temperature_c)), node('small', '', '℃'));
  $('battery-bar').value = next.vehicle.battery_percent;
  $('welcome').hidden = next.messages.length > 0;
  $('messages').replaceChildren(...next.messages.map(messageElement));
  renderAnalysis(next.last);
  $('connection-dot').classList.remove('disconnected');
  $('connection-label').textContent = '本地服务已连接';
  if (scroll) $('chat-scroll').scrollTop = $('chat-scroll').scrollHeight;
}

function updateCount() {
  $('char-count').textContent = `${$('prompt').value.length} / 2000`;
  $('send').disabled = busy || !state || !$('prompt').value.trim();
}

$('prompt').addEventListener('input', updateCount);
$('prompt').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
    event.preventDefault();
    if (!busy && state && $('prompt').value.trim()) $('chat-form').requestSubmit();
  }
});

$('chat-form').addEventListener('submit', async event => {
  event.preventDefault();
  const text = $('prompt').value.trim();
  if (!text || busy || !state) return;
  $('welcome').hidden = true;
  $('messages').append(messageElement({role:'user', content:text, at:new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}));
  $('prompt').value = '';
  updateCount();
  setBusy(true, true);
  $('chat-scroll').scrollTop = $('chat-scroll').scrollHeight;
  try {render(await request('/api/chat', {text}), true);}
  catch (error) {render(state); $('prompt').value = text; toast(error.message);}
  finally {setBusy(false); updateCount(); $('prompt').focus();}
});

document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
  $('prompt').value = button.dataset.prompt;
  updateCount();
  $('prompt').focus();
}));

async function changeSettings() {
  if (busy || !state) return;
  const settings = {mode:$('mode').value, nlu:$('nlu-mode').value};
  setBusy(true);
  try {render(await request('/api/settings', settings)); toast('运行设置已更新，车辆状态与对话已保留。');}
  catch (error) {render(state); toast(error.message);}
  finally {setBusy(false);}
}
$('mode').addEventListener('change', changeSettings);
$('nlu-mode').addEventListener('change', changeSettings);

$('reset').addEventListener('click', async () => {
  if (busy) return;
  setBusy(true);
  try {render(await request('/api/reset', {})); toast('会话已重置：电量 25%，空调 26℃。');}
  catch (error) {toast(error.message);}
  finally {setBusy(false);}
});

async function refresh(manual = false) {
  if (busy) return;
  setBusy(true);
  try {render(await request('/api/session')); if (manual) toast('会话状态已刷新。');}
  catch (error) {$('connection-dot').classList.add('disconnected'); $('connection-label').textContent = '暂未连接'; toast(error.message);}
  finally {setBusy(false);}
}
$('refresh').addEventListener('click', () => refresh(true));
refresh();
