import {createNavigation} from './navigation.js';
const $ = (id) => document.getElementById(id);
const sourceNames = {rule_nlu:'本地规则', semantic_nlu:'本地语义', nlu_guard:'边界检查', langchain_agent:'云端 Agent', planner_executor:'Planner + Executor', offline_script:'离线处理'};
const taskStatuses = {pending:'等待', running:'执行中', succeeded:'成功', failed:'失败', blocked:'已拦截', skipped:'已跳过'};
const intentNames = {set_temperature:'设置空调温度', query_vehicle:'查询车辆状态', set_headlights:'前照灯开关', adjust_seat:'调整座椅', set_wipers:'调整雨刮器', estimate_driving_range:'估算剩余续航', get_driving_history:'读取行驶记忆'};
const toolNames = {...intentNames, get_vehicle_state:'查询车辆状态', get_vehicle_location:'读取车辆位置',
  get_route_options:'查询模拟路线', find_chargers:'查询模拟充电站', search_navigation_places:'搜索高德地点',
  plan_navigation_route:'规划高德路线', navigate_to_destination:'就近导航规划'};
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
  multiple_intents:'检测到多个分句，交给Planner完整拆解，避免只执行一个子意图。',
  range_or_memory_request:'续航或长期记忆请求，交给Planner选择专用工具，避免误当普通状态查询。',
};
let state = null;
let busy = false;
let toastTimer;
let navigation;
let locationKey;
let batteryKey;
let tripSubmission = null;

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
  for (const id of ['mode', 'nlu-mode', 'reset', 'prompt', 'vehicle-position-name', 'vehicle-position-coordinates', 'vehicle-location-save', 'vehicle-location-example', 'motion-parked', 'motion-driving', 'motion-unknown', 'range-soc', 'range-capacity', 'battery-save', 'trip-distance', 'trip-energy', 'trip-save']) $(id).disabled = disabled;
  // 首次连接失败也允许手动重试，不能因尚无会话而锁住刷新按钮。
  $('refresh').disabled = busy;
  document.querySelectorAll('[data-prompt]').forEach(button => {button.disabled = disabled;});
  $('send').disabled = disabled || !$('prompt').value.trim();
  navigation?.setBusy(disabled);
  $('pending').hidden = !(busy && chat);
  if (busy && chat) $('task-plan-panel').hidden = true;
  $('pending-label').textContent = state?.mode === 'live' ? '正在识别、执行任务并整理回答…' : '正在本地识别并执行…';
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
  if (!user) header.append(node('span', 'message-source', message.error ? '调用失败' : message.safety_blocked ? '安全拦截' : message.rejected ? '操作已拒绝' : sourceNames[message.route] || '需要澄清'));
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
    case 'planner.started': return ['Planner 开始规划', `任务上限 ${d.max_tasks}`, ''];
    case 'planner.plan': return ['LLM 提交任务图', d.clarification || `${d.tasks.length} 项任务，已通过结构和依赖校验`, ''];
    case 'planner.error': return ['计划未通过校验', d.message, 'error'];
    case 'executor.plan': return ['Executor 接收任务图', `${d.tasks.length} 项任务；同一执行器按顺序处理`, ''];
    case 'task.status': return [`${d.id} · ${d.tool} · ${taskStatuses[d.status] || d.status}`, d.error?.message || `依赖：${d.depends_on.join('、') || '无'}；尝试 ${d.attempts} 次`, d.status === 'succeeded' ? 'success' : ['blocked','failed'].includes(d.status) ? 'error' : ''];
    case 'task.guard': return [`${d.task_id} · 写权限检查${d.allowed ? '通过' : '拒绝'}`, d.reason, d.allowed ? 'success' : 'error'];
    case 'task.retry': return [`${d.task_id} · 有限重试`, d.reason, ''];
    case 'memory.recorded': return ['行驶记忆已保存', d.duplicate ? '重复提交，未新增记录。' : `${d.record.distance_km}公里 / ${d.record.energy_kwh}kWh`, 'success'];
    case 'response.started': return ['正在整理回答', '根据用户问题和真实回执组织语言，不再调用工具。', ''];
    case 'response.completed': return ['回答整理完成', '实际结果已组织成面向用户的回答，任务明细仍可查看。', 'success'];
    case 'response.fallback': return ['回答整理已兜底', d.reason, ''];
    case 'executor.completed': return ['任务调度完成', `成功${d.counts.succeeded} · 拦截${d.counts.blocked} · 失败${d.counts.failed} · 跳过${d.counts.skipped}`, ''];
    case 'navigation.location': return ['读取模拟车辆位置', `${d.name} · ${d.location}`, ''];
    case 'navigation.match': return ['自动选择最近匹配地点', `${d.selected.name} · 近似直线距离 ${(d.selected.distance_m / 1000).toFixed(2)} 公里 · ${d.candidate_count} 个匹配候选`, 'success'];
    case 'safety.check': return [d.allowed ? '安全检查通过' : '车辆安全拦截', `${d.rule_id} · ${d.reason}`, d.allowed ? 'success' : 'error'];
    case 'tool.call': return [`调用 ${d.name}`, Object.keys(d.arguments).length ? JSON.stringify(d.arguments) : '无需参数', ''];
    case 'tool.result': {
      if (!d.result.ok) return ['操作被拦截', d.result.error.message, 'error'];
      const data = d.result.data;
      const summary = data.routes ? `返回 ${data.routes.length} 条${d.result.simulated ? '模拟' : '高德'}路线` : data.places ? `返回 ${data.places.length} 个高德地点` : data.chargers ? `返回 ${data.chargers.length} 个充电站` : data.status === 'near_destination' ? data.note : Object.entries(data).map(([k,v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' · ');
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
  badge.textContent = !d ? (last?.route === 'amap_navigation' ? '高德查询' : '等待输入') : last.error ? '模型出错' : last.safety_blocked ? '安全已拦截' : last.rejected ? '操作已拒绝' : d.action === 'execute' ? '本地接受' : d.action === 'clarify' ? '需要澄清' : '交给后续处理';
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
  renderVehicleLocation(next.vehicle.location);
  renderSafety(next.vehicle);
  renderDrivingMemory(next.driving_memory, next.vehicle);
  $('welcome').hidden = next.messages.length > 0;
  $('messages').replaceChildren(...next.messages.map(messageElement));
  renderAnalysis(next.last);
  renderTasks(next.execution);
  navigation?.update(next.navigation);
  $('connection-dot').classList.remove('disconnected');
  $('connection-label').textContent = '本地服务已连接';
  if (scroll) $('chat-scroll').scrollTop = $('chat-scroll').scrollHeight;
}

function renderTasks(execution) {
  const tasks = execution?.tasks || [];
  $('task-plan-panel').hidden = !tasks.length;
  if (!tasks.length) {$('task-plan-list').replaceChildren(); return;}
  const counts = execution.counts;
  $('task-counts').textContent = `成功${counts.succeeded} · 拦截${counts.blocked} · 失败${counts.failed} · 跳过${counts.skipped}`;
  $('task-plan-list').replaceChildren(...tasks.map(task => {
    const li = node('li', `task-row ${task.status}`);
    const heading = node('div', 'task-row-heading');
    heading.append(node('strong', '', `${task.id} · ${toolNames[task.tool] || task.tool}`), node('span', 'task-status', taskStatuses[task.status] || task.status));
    const dependencies = `依赖：${task.depends_on.join('、') || '无（可独立调度）'} · 尝试${task.attempts}次 · ${duration(task.duration_ms)}`;
    li.append(heading, node('p', 'task-dependencies', dependencies));
    if (task.condition) li.append(node('p', 'task-dependencies', `条件：${task.condition.ref} ${task.condition.operator} ${JSON.stringify(task.condition.value)}`));
    if (task.order_only?.length) li.append(node('p', 'task-dependencies', `其中 ${task.order_only.join('、')} 仅为执行器顺序约束，失败不会阻断后续独立任务。`));
    if (task.error) li.append(node('p', 'task-error', `${task.error.code} · ${task.error.message}`));
    const details = node('details');
    details.append(node('summary', '', '参数与实际结果'), node('pre', '', JSON.stringify({arguments:task.arguments, authorization:task.authorization, result:task.result}, null, 2)));
    li.append(details);
    return li;
  }));
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
  finally {setBusy(false); updateCount(); if (!$('chat-workspace').hidden) $('prompt').focus();}
});

document.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => {
  $('prompt').value = button.dataset.prompt;
  updateCount();
  $('prompt').focus();
  const hints = [];
  if (button.dataset.live && state?.mode !== 'live') hints.push('请先将左侧对话模式切换为“真实模型 Agent”');
  if (button.dataset.requiresLocation && !state?.vehicle.location) hints.push('请在右侧保存模拟车辆位置');
  if (hints.length) toast(`示例已填入，尚未发送。${hints.join('；')}。`);
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
  try {render(await request('/api/reset', {})); navigation.reset(); toast('会话已重置：电量 25%，空调 26℃。');}
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

function renderSafety(vehicle) {
  const motion = vehicle.motion || {};
  $('motion-status').textContent = motion.label || '状态未知';
  $('motion-status').className = `mini-badge ${motion.status === 'parked' ? 'accepted' : 'warning'}`;
  $('motion-speed').textContent = typeof motion.speed_kph === 'number' ? `${motion.speed_kph} km/h` : '未知';
  $('motion-gear').textContent = motion.gear || '未知';
  $('motion-brake').textContent = motion.parking_brake === true ? '已启用' : motion.parking_brake === false ? '未启用' : '未知';
  $('headlights-state').textContent = vehicle.headlights_on ? '已打开' : '已关闭';
  $('seat-state').textContent = `${vehicle.seat_position_percent ?? '—'}%`;
  $('wipers-state').textContent = {off:'关闭', low:'低速', high:'高速'}[vehicle.wiper_mode] || '未知';
  for (const [preset, status] of [['parked','parked'], ['driving','moving'], ['unknown','unknown']]) {
    $(`motion-${preset}`).setAttribute('aria-pressed', String(motion.status === status));
  }
  const decision = vehicle.last_safety_decision;
  $('safety-last').className = `safety-verdict${decision && !decision.allowed ? ' blocked' : ''}`;
  $('safety-last').textContent = decision ? `最近判定 · ${decision.reason}` : '还没有执行安全判定';
}

const motionPresets = {
  parked:{speed_kph:0, gear:'P', parking_brake:true},
  driving:{speed_kph:40, gear:'D', parking_brake:false},
  unknown:{speed_kph:null, gear:null, parking_brake:null},
};
for (const [name, values] of Object.entries(motionPresets)) {
  $(`motion-${name}`).addEventListener('click', async () => {
    if (busy || !state) return;
    setBusy(true);
    try {
      render(await request('/api/vehicle/driving-state', values));
      toast('模拟行驶状态已更新；已拒绝的动作不会自动重试。');
    } catch (error) {toast(error.message);}
    finally {setBusy(false);}
  });
}

function renderVehicleLocation(location) {
  $('vehicle-location-summary').textContent = location ? `${location.name} · ${location.location}（模拟）` : '尚未设置模拟车辆位置';
  const nextKey = location ? JSON.stringify(location) : '';
  if (nextKey !== locationKey) {
    locationKey = nextKey;
    $('vehicle-position-name').value = location?.name || '';
    $('vehicle-position-coordinates').value = location?.location || '';
    $('vehicle-location-settings').open = !location;
  }
}

$('vehicle-location-example').addEventListener('click', () => {
  $('vehicle-position-name').value = '北京天安门（模拟）';
  $('vehicle-position-coordinates').value = '116.397428,39.909230';
  toast('已填入公开地点示例，请点击“保存位置”生效。');
});
$('vehicle-location-save').addEventListener('click', async () => {
  if (busy || !state) return;
  setBusy(true);
  try {
    render(await request('/api/vehicle/location', {
      name:$('vehicle-position-name').value.trim(), location:$('vehicle-position-coordinates').value.trim(),
    }));
    toast('模拟车辆位置已保存，旧路线已清空。可在真实模型模式说“导航到北京南站”。');
  } catch (error) {toast(error.message);}
  finally {setBusy(false);}
});

function renderDrivingMemory(memory, vehicle) {
  if (!memory?.available) {
    $('range-value').textContent = '—';
    $('range-badge').textContent = '记忆不可用';
    $('range-basis').textContent = memory?.error || '尚未读取行驶记忆。';
    $('driving-memory-list').replaceChildren();
    return;
  }
  const e = memory.estimate;
  $('range-value').textContent = String(e.estimated_range_km);
  $('range-badge').textContent = e.basis === 'history' ? `${e.sample_count}条历史` : '默认假设';
  $('range-basis').textContent = `当前${e.battery_percent}% · 容量${e.capacity_kwh}kWh · ${e.consumption_kwh_per_100km.toFixed(2)}kWh/100km。` +
    (e.basis === 'history' ? `最近${e.sample_count}条合计${e.historical_distance_km}公里，按里程加权。` : '尚无历史，使用18kWh/100km教学能耗。') +
    (e.capacity_source === 'teaching_default' ? '容量60kWh尚为教学假设。' : '') +
    (e.sample_count > 0 && e.sample_count < 3 ? '历史少于3条，样本不足。' : '');
  const key = `${vehicle.battery_percent}|${memory.capacity_kwh}`;
  if (key !== batteryKey) {
    batteryKey = key;
    $('range-soc').value = String(vehicle.battery_percent);
    $('range-capacity').value = String(memory.capacity_kwh);
  }
  const rows = memory.records.length ? memory.records.map(row => node('li', '', `#${row.id} · ${row.distance_km}公里 / ${row.energy_kwh}kWh（${row.consumption_kwh_per_100km.toFixed(2)}kWh/100km）`)) : [node('li', '', '还没有保存的行程。')];
  $('driving-memory-list').replaceChildren(...rows);
}

$('battery-save').addEventListener('click', async () => {
  if (busy || !state || !$('range-soc').reportValidity() || !$('range-capacity').reportValidity()) return;
  setBusy(true);
  try {
    render(await request('/api/vehicle/battery', {battery_percent:Number($('range-soc').value), capacity_kwh:Number($('range-capacity').value)}));
    toast('模拟电量已更新，本车可用容量已长期保存，续航已重新计算。');
  } catch (error) {toast(error.message);}
  finally {setBusy(false);}
});

for (const id of ['trip-distance','trip-energy']) $(id).addEventListener('input', () => {tripSubmission = null;});
$('trip-save').addEventListener('click', async () => {
  if (busy || !state || !$('trip-distance').reportValidity() || !$('trip-energy').reportValidity()) return;
  const values = {distance_km:Number($('trip-distance').value), energy_kwh:Number($('trip-energy').value)};
  // 连接超时后重试同一次保存继续使用原ID，服务端按车辆＋ID去重。
  if (!tripSubmission) tripSubmission = {...values, record_id:crypto.randomUUID()};
  setBusy(true);
  try {
    const result = await request('/api/driving-memory/trips', tripSubmission);
    render(result.session);
    tripSubmission = null;
    $('trip-distance').value = '';
    $('trip-energy').value = '';
    toast(result.result.duplicate ? '该行程此前已保存，未重复计入。' : '行程已保存到本车长期记忆，续航已重新计算；当前电量未扣减。');
  } catch (error) {toast(error.message);}
  finally {setBusy(false);}
});

navigation = createNavigation({request, onSession:render, notify:toast,
  acquire:() => {if (busy || !state) return false; setBusy(true); return true;},
  release:() => setBusy(false),
});
refresh();
