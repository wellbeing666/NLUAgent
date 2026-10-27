// 导航面板只调用本地API；高德Key和原始路线几何留在Python服务端。
export function createNavigation({request, onSession, acquire, release, notify}) {
  const $ = id => document.getElementById(id);
  let disabled = true;
  let configured = false;
  let selected = {origin:null, destination:null};
  let plan = null;
  let planKey = '';
  let activeRoute = null;
  let imageVersion = 0;

  function element(tag, className, text) {
    const el = document.createElement(tag);
    el.className = className;
    el.textContent = text;
    return el;
  }

  function openView(navigation) {
    $('navigation-workspace').hidden = !navigation;
    $('chat-workspace').hidden = navigation;
    $('tab-chat').classList.toggle('nav-active', !navigation);
    $('tab-navigation').classList.toggle('nav-active', navigation);
    $('tab-chat').setAttribute('aria-pressed', String(!navigation));
    $('tab-navigation').setAttribute('aria-pressed', String(navigation));
  }
  $('tab-chat').addEventListener('click', () => openView(false));
  $('tab-navigation').addEventListener('click', () => openView(true));

  function message(text, error = false) {
    $('navigation-message').textContent = text;
    $('navigation-message').classList.toggle('error', error);
  }

  function controls() {
    for (const side of ['origin', 'destination']) {
      for (const suffix of ['city', 'keyword', 'search']) $(`${side}-${suffix}`).disabled = disabled;
    }
    $('navigation-strategy').disabled = disabled;
    $('plan-navigation').disabled = disabled || !selected.origin || !selected.destination;
    document.querySelectorAll('.place-option, .route-option').forEach(button => {button.disabled = disabled;});
  }

  function hideResult() {
    imageVersion++;
    activeRoute = null;
    $('navigation-result').hidden = true;
    $('navigation-empty').hidden = false;
    $('navigation-map').hidden = true;
    $('navigation-map').removeAttribute('src');
    // 起终点编辑后禁用旧链接，避免将上次路线当作新路线。
    $('navigation-web-link').removeAttribute('href');
    $('navigation-app-link').removeAttribute('href');
  }

  function pointOnly(place) {
    return {name:place.name, location:place.location, ...(place.id ? {id:place.id} : {})};
  }

  async function search(side) {
    if (!acquire()) return;
    selected[side] = null;
    hideResult();
    $(`${side}-results`).replaceChildren();
    $(`${side}-selected`).textContent = '尚未选择具体地点';
    message('正在查询高德地点…');
    try {
      const response = await request('/api/navigation/search', {
        keywords:$(`${side}-keyword`).value.trim(), city:$(`${side}-city`).value.trim(),
      });
      onSession(response.session);
      if (!response.result.ok) throw new Error(response.result.error.message);
      const places = response.result.data.places;
      for (const place of places) {
        const button = element('button', 'place-option', '');
        button.type = 'button';
        button.append(element('strong', '', place.name), element('span', '', place.address || place.location));
        button.addEventListener('click', () => {
          selected[side] = pointOnly(place);
          $(`${side}-selected`).textContent = `已选：${place.name} · ${place.location}`;
          $(`${side}-results`).replaceChildren();
          message('地点已选择，确认起终点后查询路线。');
          hideResult();
          controls();
        });
        $(`${side}-results`).append(button);
      }
      message(places.length ? `找到 ${places.length} 个候选，请点击具体地点。` : '未找到地点，请检查城市或补全地点名称。');
    } catch (error) {message(error.message, true);}
    finally {release();}
  }

  for (const side of ['origin', 'destination']) {
    $(`${side}-search`).addEventListener('click', () => search(side));
    for (const suffix of ['keyword', 'city']) {
      $(`${side}-${suffix}`).addEventListener('input', () => {
        selected[side] = null;
        $(`${side}-results`).replaceChildren();
        $(`${side}-selected`).textContent = '输入已变化，请重新搜索并选择地点';
        hideResult();
        controls();
      });
      $(`${side}-${suffix}`).addEventListener('keydown', event => {
        if (event.key === 'Enter' && !event.isComposing) {event.preventDefault(); search(side);}
      });
    }
  }
  $('navigation-strategy').addEventListener('change', hideResult);

  async function loadMap(route) {
    const version = ++imageVersion;
    $('navigation-map').hidden = true;
    $('navigation-map-message').hidden = false;
    $('navigation-map-message').textContent = '正在加载高德地图概览…';
    $('retry-navigation-map').hidden = true;
    try {
      const response = await request('/api/navigation/map', {route_id:route.route_id});
      if (version !== imageVersion) return;
      if (typeof response.image !== 'string' || !/^data:image\/(png|jpeg);base64,/.test(response.image)) throw new Error('地图图片格式不正确。');
      $('navigation-map').src = response.image;
      $('navigation-map').hidden = false;
      $('navigation-map-message').hidden = true;
    } catch (error) {
      if (version !== imageVersion) return;
      $('navigation-map-message').textContent = `${error.message} 路线与路书仍可查看。`;
      $('retry-navigation-map').hidden = false;
    }
  }

  function selectRoute(route) {
    activeRoute = route;
    document.querySelectorAll('.route-option').forEach(button => {
      const active = button.dataset.routeId === route.route_id;
      button.classList.toggle('selected', active);
      button.setAttribute('aria-pressed', String(active));
    });
    $('navigation-steps').replaceChildren(...route.steps.map(step => element('li', '', step.instruction || `${step.road} · ${Math.round(step.distance_m)}米`)));
    if (!route.steps.length) $('navigation-steps').append(element('li', '', '该路线未返回分段路书，请在高德查看。'));
    loadMap(route);
  }

  function showPlan(next) {
    hideResult();
    plan = next;
    $('navigation-empty').hidden = true;
    $('navigation-result').hidden = false;
    $('navigation-title').textContent = `${plan.origin.name} → ${plan.destination.name}`;
    const selection = plan.selection;
    $('navigation-selection').hidden = !selection;
    $('navigation-selection').textContent = selection
      ? `从手动设置的模拟车辆位置出发，已自动选择“${selection.selected.name}”。近似直线距离 ${(selection.selected.distance_m / 1000).toFixed(2)} 公里；本次 ${selection.candidate_count} 个名称匹配候选中最近。${selection.search_scope === 'national_candidates' ? '周边未命中，使用全国搜索返回的候选。' : '候选来自周边50公里搜索。'} 距离不是实际驾车距离。`
      : '';
    // 只允许后端固定构造的高德HTTPS地址，禁止任意协议或外部主机链接。
    for (const [id, url] of [['navigation-web-link', plan.web_url], ['navigation-app-link', plan.app_url]]) {
      const parsed = new URL(url);
      if (parsed.protocol === 'https:' && parsed.hostname === 'uri.amap.com' && parsed.pathname === '/navigation') $(id).href = parsed.href;
    }
    $('navigation-routes').replaceChildren(...plan.routes.map((route, index) => {
      const button = element('button', 'route-option', '');
      button.type = 'button';
      button.dataset.routeId = route.route_id;
      button.append(element('strong', '', `${index + 1}. ${route.name}`),
        element('span', '', `${route.duration_minutes} 分钟 · ${route.distance_km} 公里`),
        element('small', '', `预计过路费 ¥${route.tolls_yuan}`));
      button.addEventListener('click', () => selectRoute(route));
      return button;
    }));
    if (plan.routes.length) selectRoute(plan.routes[0]);
    controls();
  }

  $('retry-navigation-map').addEventListener('click', () => {if (activeRoute) loadMap(activeRoute);});
  $('plan-navigation').addEventListener('click', async () => {
    if (!selected.origin || !selected.destination || !acquire()) return;
    hideResult();
    message('正在查询高德驾车路线…');
    try {
      const response = await request('/api/navigation/route', {
        origin:selected.origin, destination:selected.destination, strategy:$('navigation-strategy').value,
      });
      onSession(response.session);
      if (!response.result.ok) throw new Error(response.result.error.message);
      message(`已取得 ${response.result.data.routes.length} 条真实路线。尚未启动导航。`);
    } catch (error) {message(error.message, true);}
    finally {release();}
  });

  return {
    setBusy(value) {disabled = value; controls();},
    update(navigation) {
      configured = !!navigation?.configured;
      $('amap-status').textContent = configured ? 'Web服务 Key 已配置' : '等待配置 Key';
      $('amap-config-hint').hidden = configured;
      const next = navigation?.plan;
      const nextKey = next?.routes?.map(route => route.route_id).join(',') || '';
      if (nextKey !== planKey) {
        planKey = nextKey;
        if (next) {
          selected = {origin:pointOnly(next.origin), destination:pointOnly(next.destination)};
          $('navigation-strategy').value = next.strategy;
          for (const side of ['origin', 'destination']) {
            $(`${side}-selected`).textContent = `已选：${selected[side].name} · ${selected[side].location}`;
            $(`${side}-keyword`).value = selected[side].name;
            $(`${side}-results`).replaceChildren();
          }
          showPlan(next);
          if (next.automatic) {
            openView(true);
            message(`对话已自动选择“${next.destination.name}”并规划路线，可查看地图与路书。尚未启动外部导航。`);
          }
          notify('高德路线已更新，可在“高德导航”查看。');
        } else {plan = null; hideResult();}
      }
    },
    reset() {
      selected = {origin:null, destination:null};
      planKey = '';
      plan = null;
      for (const side of ['origin', 'destination']) {
        $(`${side}-keyword`).value = '';
        $(`${side}-results`).replaceChildren();
        $(`${side}-selected`).textContent = '尚未选择具体地点';
      }
      hideResult();
      message('会话已重置，从搜索起点开始。');
      controls();
    },
  };
}
