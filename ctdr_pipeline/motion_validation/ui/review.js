'use strict';

function nativeXY(clientX, clientY, rect, width, height) {
  if (!(rect.width > 0 && rect.height > 0)) return null;
  const x = (clientX - rect.left) * width / rect.width, y = (clientY - rect.top) * height / rect.height;
  return x >= 0 && x < width && y >= 0 && y < height ? [x, y] : null;
}

function putObservation(point, frame, state, xy) {
  const observation = {frame_index: frame, state, xy_px: state === 'visible' ? xy : null, reviewed: true};
  point.observations = point.observations.filter(o => o.frame_index !== frame).concat(observation).sort((a,b) => a.frame_index-b.frame_index);
  if (point.decision === 'include' && !point.observations.some(o => o.state === 'visible')) point.decision = 'pending';
}

function checkReview(doc, manifest) {
  const fail = message => {throw new Error(message);};
  if (!doc || doc.pack_id !== manifest.pack_id || doc.case_id !== manifest.case_id || doc.schema_version !== 1 || doc.research_only !== true) fail('复核文件不属于当前数据包或版本。');
  if (!['draft','submitted'].includes(doc.review_status) || typeof doc.reviewer !== 'string' || (doc.review_status === 'submitted' && !doc.reviewer.trim())) fail('请填写复核人编号。');
  if (!Array.isArray(doc.runs) || doc.runs.length !== manifest.runs.length) fail('序列列表不匹配。');
  const runIds = new Set();
  for (const run of doc.runs) {
    const source = manifest.runs.find(r => r.run_id === run.run_id);
    if (!source || runIds.has(run.run_id) || !Array.isArray(run.points)) fail('未知/重复序列。');
    runIds.add(run.run_id);
    const ids = new Set(), sources = new Set(), [frames, rows, columns] = source.shape_frame_row_column;
    for (const p of run.points) {
      if (typeof p.point_id !== 'string' || !/^[A-Za-z0-9_-]{1,80}$/.test(p.point_id) || ids.has(p.point_id)) fail('点编号无效或重复。');
      ids.add(p.point_id);
      if (p.origin === 'automatic') {
        if (!Number.isInteger(p.source_track_id) || !source.source_track_ids.includes(p.source_track_id) || p.point_id !== `auto_${p.source_track_id}`) fail('候选点来源不匹配。');
        sources.add(p.source_track_id);
      } else if (p.origin !== 'manual' || p.source_track_id !== null || !p.point_id.startsWith('manual_')) fail('人工点来源不匹配。');
      if (!['pending','include','exclude'].includes(p.decision) || typeof p.anatomical_label !== 'string' || !Array.isArray(p.observations)) fail('点记录无效。');
      const reasons = ['catheter','background','ambiguous','tracking_error','other'];
      if (p.exclusion_reason !== null && !reasons.includes(p.exclusion_reason)) fail('排除原因无效。');
      if (p.decision === 'exclude' && !p.exclusion_reason) fail('排除点需填写原因。');
      const seenFrames = new Set(); let visible = 0;
      for (const o of p.observations) {
        if (!Number.isInteger(o.frame_index) || o.frame_index < 0 || o.frame_index >= frames || seenFrames.has(o.frame_index) || o.reviewed !== true) fail('帧编号无效、重复或未明确复核。');
        seenFrames.add(o.frame_index);
        if (o.state === 'visible') {
          if (!Array.isArray(o.xy_px) || o.xy_px.length !== 2 || !o.xy_px.every(v => typeof v === 'number' && Number.isFinite(v)) || o.xy_px[0] < 0 || o.xy_px[0] >= columns || o.xy_px[1] < 0 || o.xy_px[1] >= rows) fail('坐标超出原图或不是有限数值。');
          visible++;
        } else if (!['not_visible','uncertain'].includes(o.state) || o.xy_px !== null) fail('不可见/不确定点应保留空坐标。');
      }
      if (p.decision === 'include' && (!doc.reviewer.trim() || !p.anatomical_label.trim() || p.exclusion_reason !== null || !visible)) fail('纳入点需复核人、解剖名称、至少一帧明确坐标，且不能同时排除。');
    }
    if (sources.size !== source.source_track_ids.length) fail('不能静默删除自动候选，请明确选择排除。');
  }
  return true;
}

if (typeof module !== 'undefined' && module.exports) module.exports = {nativeXY, putObservation, checkReview};
if (typeof window !== 'undefined' && window.REVIEW_DATA) (() => {
  const data = window.REVIEW_DATA, manifest = data.manifest, $ = id => document.getElementById(id);
  let doc = structuredClone(data.template), runId = manifest.runs[0].run_id, frame = 0, pointId = null, mode = 'select', timer = null, renderToken = 0, rendered = null;
  const canvas = $('image'), context = canvas.getContext('2d');
  const bound = () => manifest.runs.find(r => r.run_id === runId);
  const currentRun = () => doc.runs.find(r => r.run_id === runId);
  const currentPoint = () => currentRun().points.find(p => p.point_id === pointId);
  const autoTrack = p => p?.origin === 'automatic' ? data.observations[runId].tracks.find(t => t.track_id === p.source_track_id) : null;
  const autoXY = p => {const t = autoTrack(p); return t?.valid[frame] ? t.xy_px[frame] : null;};
  const notify = text => {$('message').textContent = text;};
  const stop = () => {clearInterval(timer); timer = null; $('play').textContent = '播放';};
  const annotationReady = () => {
    const ready = rendered && rendered.runId === runId && rendered.frame === frame && rendered.token === renderToken;
    if (!ready) notify('帧加载中或失败，禁止将旧画面标记到新帧；请等图像加载完成。');
    return ready;
  };
  function setMode(value) {mode = value; $('mark').classList.toggle('active', value === 'mark'); $('add').classList.toggle('active', value === 'add');}
  function refreshList() {
    const labels = {pending:'待复核',include:'纳入',exclude:'排除'};
    $('points').replaceChildren(...currentRun().points.map(p => {
      const option = document.createElement('option'), t = autoTrack(p);
      option.value = p.point_id; option.textContent = `${p.point_id} · ${labels[p.decision]} · 自动覆盖 ${t ? t.valid.filter(Boolean).length : 0}/${bound().shape_frame_row_column[0]}`;
      return option;
    }));
    if (!currentPoint()) pointId = currentRun().points[0]?.point_id ?? null;
    $('points').value = pointId ?? '';
  }
  function pointForm() {
    const p = currentPoint();
    $('pointInfo').textContent = p ? `选中 ${p.point_id}；人工确认 ${p.observations.length} 帧` : '尚无点，可新增人工点';
    $('label').value = p?.anatomical_label ?? ''; $('decision').value = p?.decision ?? 'pending'; $('reason').value = p?.exclusion_reason ?? '';
    $('marks').textContent = p?.observations.map(o => `帧 ${o.frame_index}：${o.state}${o.xy_px ? ' ['+o.xy_px.map(v => v.toFixed(1)).join(', ')+']' : ''}`).join('； ') ?? '';
  }
  async function render() {
    const token = ++renderToken, r = bound(), [count, rows, columns] = r.shape_frame_row_column;
    rendered = null; canvas.dataset.loading = 'true'; canvas.setAttribute('aria-busy','true');
    $('frame').max = count-1; $('frame').value = frame;
    const time = data.observations[runId].frame_times_ms?.[frame];
    $('frameLabel').textContent = `${frame}/${count-1}（0 起始）${Number.isFinite(time) ? ' · '+time.toFixed(1)+' ms' : ' · 采集时间未知'}`;
    const image = new Image(); image.src = `frames/${runId}/${String(frame).padStart(4,'0')}.png`;
    try {await image.decode();} catch (_) {if (token === renderToken) notify('帧加载失败，请检查数据包完整性。'); return;}
    if (token !== renderToken) return;
    canvas.width = columns; canvas.height = rows;
    context.drawImage(image, 0, 0);
    const dot = (xy, color, radius, text) => {context.strokeStyle=color; context.lineWidth=1.5; context.beginPath(); context.arc(xy[0],xy[1],radius,0,2*Math.PI); context.stroke(); if(text){context.fillStyle=color; context.font='10px sans-serif'; context.fillText(text,xy[0]+5,xy[1]-5);}};
    if ($('showAuto').checked) for (const p of currentRun().points) {
      const xy = autoXY(p); if (xy) dot(xy, p.decision === 'exclude' ? '#bf7a7a' : '#9aa9b5', 2.4, '');
    }
    for (const p of currentRun().points) {
      const o = p.observations.find(o => o.frame_index === frame && o.state === 'visible');
      if (o) dot(o.xy_px, '#59edc6', 4, p.point_id);
    }
    const p = currentPoint(), xy = autoXY(p); if (xy) dot(xy, '#ffdb64', 6, p.point_id);
    rendered = {runId, frame, token}; canvas.dataset.frameIndex=String(frame); canvas.dataset.runId=runId;
    canvas.dataset.loading='false'; canvas.setAttribute('aria-busy','false');
  }
  function selectSeries() {
    stop(); setMode('select'); frame = bound().reference_frame_index; pointId = null;
    const g = bound().geometry;
    $('geometry').textContent = `本序列标定接受：否\n旧报告状态：${g.quality_status}\n转移来源：${g.reference_series ?? '未知'}\n来源序列 RMS：${g.reference_series_rms_px ?? '未知'} px（不是当前序列误差）\n当前序列实测重投影 RMS：未取得\n基底正交误差：${g.basis_orthogonality_error}\n基底行列式：${g.basis_determinant}\nSID 距离误差：${g.sdd_distance_error_mm} mm\nSOD 距离误差：${g.sod_distance_error_mm} mm\n数学自洽不代表解剖对齐。需独立 CT–XA 对应点验证，当前不开放患者三维优化。`;
    refreshList(); pointForm(); render();
  }
  function record(state, xy) {
    if (!annotationReady()) return;
    const p = currentPoint(); if (!p) {notify('请先选择或新增一个点。'); return;}
    const displayedFrame = rendered.frame;
    putObservation(p, displayedFrame, state, xy); refreshList(); pointForm(); render(); notify(`已记录 ${p.point_id} 帧 ${displayedFrame}：${state}；其他帧未自动确认。`);
  }
  canvas.addEventListener('click', event => {
    stop(); if (!annotationReady()) return;
    const xy = nativeXY(event.clientX, event.clientY, canvas.getBoundingClientRect(), canvas.width, canvas.height); if (!xy) return;
    if (mode === 'add') {
      let id = 1; while (currentRun().points.some(p => p.point_id === `manual_${id}`)) id++;
      const p = {point_id:`manual_${id}`,origin:'manual',source_track_id:null,decision:'pending',anatomical_label:'',exclusion_reason:null,observations:[]};
      currentRun().points.push(p); pointId=p.point_id; putObservation(p,frame,'visible',xy); setMode('select'); refreshList(); pointForm(); render(); notify('已新增当前帧人工点；请填写名称并决定是否纳入。');
    } else if (mode === 'mark') {record('visible',xy); setMode('select');}
    else {
      const choices=currentRun().points.map(p => {const o=p.observations.find(o=>o.frame_index===frame&&o.state==='visible');const coord=o?.xy_px??autoXY(p);return {p,d:coord?Math.hypot(coord[0]-xy[0],coord[1]-xy[1]):Infinity};}).sort((a,b)=>a.d-b.d);
      if(choices[0]?.d < 15){pointId=choices[0].p.point_id;$('points').value=pointId;pointForm();render();}
      else notify('附近没有候选，请使用“新增人工点”。');
    }
  });
  $('series').replaceChildren(...manifest.runs.map(r => {const o=document.createElement('option');o.value=r.run_id;o.textContent=r.run_id;return o;}));
  $('case').textContent=`病例 ${manifest.case_id} · ${manifest.runs.length} 个序列 · 数据包 ${manifest.pack_id.slice(0,16)}…`;
  $('series').onchange=()=>{runId=$('series').value;selectSeries();};
  $('frame').oninput=()=>{stop();frame=Number($('frame').value);render();};
  const step=delta=>{stop();frame=Math.max(0,Math.min(bound().shape_frame_row_column[0]-1,frame+delta));render();};
  $('prev').onclick=()=>step(-1); $('next').onclick=()=>step(1); $('reference').onclick=()=>{stop();frame=bound().reference_frame_index;render();};
  $('play').onclick=()=>{if(timer){stop();return;}timer=setInterval(()=>{frame=(frame+1)%bound().shape_frame_row_column[0];render();},180);$('play').textContent='暂停（仅预览速度）';};
  $('showAuto').onchange=render; $('points').onchange=()=>{pointId=$('points').value;setMode('select');pointForm();render();};
  $('add').onclick=()=>{stop();setMode('add');notify('请在图像上点击一个可确认的血管特征点。');};
  $('mark').onclick=()=>{stop();setMode('mark');notify('请在图像上点击当前帧的正确位置。');};
  $('confirm').onclick=()=>{stop();const xy=autoXY(currentPoint());if(xy)record('visible',[...xy]);else notify('当前帧没有有效建议，请点击修正或标为不可见。');};
  $('invisible').onclick=()=>{stop();record('not_visible',null);}; $('uncertain').onclick=()=>{stop();record('uncertain',null);};
  $('removeMark').onclick=()=>{if(!annotationReady())return;const p=currentPoint();if(p){p.observations=p.observations.filter(o=>o.frame_index!==rendered.frame);if(p.decision==='include'&&!p.observations.some(o=>o.state==='visible'))p.decision='pending';pointForm();refreshList();render();}};
  $('label').oninput=()=>{const p=currentPoint();if(p){p.anatomical_label=$('label').value;if(!p.anatomical_label.trim()&&p.decision==='include'){p.decision='pending';$('decision').value='pending';refreshList();}}};
  $('reviewer').oninput=()=>{doc.reviewer=$('reviewer').value;};
  function changeDecision() {
    const p=currentPoint();if(!p)return;
    p.decision=$('decision').value;p.exclusion_reason=$('reason').value||null;
    try{checkReview(doc,manifest);}catch(error){p.decision='pending';p.exclusion_reason=null;notify(error.message);}
    refreshList();pointForm();render();
  }
  $('decision').onchange=()=>{if($('decision').value==='exclude'&&!$('reason').value)$('reason').value='ambiguous';if($('decision').value==='include')$('reason').value='';changeDecision();};
  $('reason').onchange=()=>{if($('reason').value)$('decision').value='exclude';else if($('decision').value==='exclude')$('decision').value='pending';changeDecision();};
  function download(status) {
    stop();doc.reviewer=$('reviewer').value;const output=structuredClone(doc);output.review_status=status;
    try{checkReview(output,manifest);}catch(error){notify(error.message);return;}
    const url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'})), link=document.createElement('a');
    link.href=url;link.download=`heart_motion_review_${manifest.case_id}_${status}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    notify('已下载；仅明确标记的帧被确认。Python 导入时会进一步核验源数据。');
  }
  $('draft').onclick=()=>download('draft'); $('submit').onclick=()=>download('submitted');
  $('import').onchange=async()=>{stop();const file=$('import').files[0];if(!file)return;try{const imported=JSON.parse(await file.text());checkReview(imported,manifest);doc=imported;$('reviewer').value=doc.reviewer;selectSeries();notify('已导入同一数据包的复核；尚未进行新的追踪或三维优化。');}catch(error){notify('拒绝导入：'+error.message);}finally{$('import').value='';}};
  selectSeries();
})();
