const SID='riverside-flood-01',$=s=>document.querySelector(s),sess='web-'+Math.random().toString(36).slice(2,8);
const COL={3:'#2563eb',2:'#e5484d',1:'#f5a524'},esc=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let area,S,snaps=[],map,lay,tiles={},timer=null,aiOK=true,lastChanges=0;
const BASE=location.port==='5000'?'':'http://127.0.0.1:5000';
async function api(p,body){const r=await fetch(BASE+'/api/'+p,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});
 const j=await r.json().catch(()=>({}));if(!r.ok)throw Object.assign(new Error(j.error?.message||r.status),{status:r.status});return j}
const dLat=()=>(area.bounds.max_lat-area.bounds.min_lat)/area.grid_height,dLon=()=>(area.bounds.max_lon-area.bounds.min_lon)/area.grid_width;
const ll=(r,c)=>[area.bounds.max_lat-(r+.5)*dLat(),area.bounds.min_lon+(c+.5)*dLon()];
const rect=(r,c)=>[[area.bounds.max_lat-(r+1)*dLat(),area.bounds.min_lon+c*dLon()],[area.bounds.max_lat-r*dLat(),area.bounds.min_lon+(c+1)*dLon()]];
const pin=(cls,t)=>L.divIcon({className:'',html:`<div class="mk ${cls}">${t}</div>`,iconSize:[26,26],iconAnchor:[13,13]});
const haz=()=>{const v=$('#view').value;return v==='live'?S.hazards:snaps.find(h=>h.hours==v)||S.hazards};

function drawMap(){
 lay.clearLayers();const H=haz();
 H.risk_grid.forEach((row,r)=>row.forEach((v,c)=>v&&L.rectangle(rect(r,c),{stroke:false,fillColor:COL[v],fillOpacity:.5}).bindTooltip(['','Medium risk','High risk','Flooded – blocked'][v]).addTo(lay)));
 S.walls.forEach(([r,c])=>L.rectangle(rect(r,c),{color:'#475569',weight:1,fillColor:'#94a3b8',fillOpacity:.9}).bindTooltip('Building').addTo(lay));
 const ev=[...S.events].reverse().find(e=>e.changed&&e.trigger!=='start'&&(e.previous_route||[]).length>1);
 if(ev)L.polyline(ev.previous_route.map(p=>ll(...p)),{color:'#e5484d',weight:4,dashArray:'6 8'}).bindTooltip('Blocked route').addTo(lay);
 const a=S.agents[0];L.polyline(a.trail.map(p=>ll(...p)),{color:'#93c5fd',weight:3}).addTo(lay);
 S.routes.filter(r=>r.status==='OPEN').forEach(r=>L.polyline(r.path.map(p=>ll(...p)),{color:'#16a34a',weight:5}).bindTooltip('Optimal route').addTo(lay));
 S.victims.forEach(v=>L.marker(ll(...v.position),{icon:v.rescued?pin('ok','✓'):pin(v.priority==='critical'?'crit':'nor','!')}).bindTooltip(v.rescued?'Rescued':v.priority==='critical'?'Critical person':'Person to rescue').addTo(lay));
 L.marker(ll(...S.exit.position),{icon:pin('ok','S')}).bindTooltip('Safe zone / exit').addTo(lay);
 L.marker(ll(...a.position),{icon:pin('bot','R'),zIndexOffset:900}).bindTooltip('Rescue robot – '+a.state).addTo(lay);
}
function panels(){
 const H=haz(),e=H.exposure,p=e.estimated_people_at_risk,m=S.metrics,hrs=H.hours;
 const lv=p>=500?['High','hi']:p>0?['Medium','md']:['Low','lo'];
 $('#banner').hidden=$('#view').value==='live';$('#banner').textContent='Preview: flood at hour '+hrs;
 $('#riskb').innerHTML=`<div class="lv ${lv[1]}"><svg class="i"><use href="#drop"/></svg><div>${lv[0]} risk<small>${esc(area.name)}</small></div></div>
 <div class="kv"><div><small>Water level</small><b>${Math.min(4,hrs*.4).toFixed(1)} m</b></div><div><small>People at risk</small><b>${p}</b></div>
 <div><small>Flooded cells</small><b>${e.blocked_cells}</b></div><div><small>Elevated-risk cells</small><b>${e.low_risk_cells+e.medium_risk_cells}</b></div></div>
 <button class="btn g w" id="det">View detailed risk data</button>`;
 $('#det').onclick=()=>{dlgb.innerHTML=`<h3>Risk grid · hour ${hrs}</h3><div class="cells">${H.risk_grid.flat().map(v=>`<span style="background:${COL[v]||'#16a34a55'}">${v}</span>`).join('')}</div>
  <small>0 safe · 1 medium · 2 high · 3 flooded (blocked). Confidence ${Math.round((H.confidence||0)*100)}%.</small><ul>${S.assumptions.map(a=>`<li>${esc(a)}</li>`).join('')}</ul><small>${esc(S.disclaimer)}</small>`;dlg.showModal()};
 const rt=S.routes.find(r=>r.status==='OPEN'),going=!!timer,done=['DONE','FAILED'].includes(S.status);
 $('#planb').innerHTML=`<div class="row"><span>People to rescue</span><b>${m.victims_remaining} of ${m.victims_total}</b></div>
 <div class="row"><span>Rescue robots</span><b>${S.agents.length}</b></div>
 <div class="row"><span>Next route</span><b>${rt?rt.cost+' cost · '+(rt.path.length-1)+' cells':'—'}</b></div>
 <div class="row"><span>Estimated time</span><b>${rt?'~'+((rt.path.length-1)/2)+' h (simulated)':'—'}</b></div>
 <div class="row"><span>Method</span><b>CSP + Risk-aware A*</b></div>
 <button class="btn w" id="go" style="margin-top:10px">${done?'Restart simulation':going?'Pause simulation':'Start simulation'}</button>`;
 $('#go').onclick=()=>done?reset().then(play):going?stop():play();
 const c=m.last_comparison,a=c&&c.risk_astar,b=c&&c.bfs;
 $('#simb').innerHTML=`<div class="row"><span>Status</span><span class="tag">${S.status}</span></div>
 <div class="row"><span>People rescued</span><b>${m.victims_rescued} / ${m.victims_total}</b></div>
 <div class="row"><span>Flood hour</span><b>${S.hours} of 10</b></div>
 <div class="bar"><i style="width:${Math.min(100,S.hours*10)}%"></i></div>
 <div class="row"><span>Route changes</span><b>${m.route_changes}</b></div>
 <div class="row"><span>A* vs BFS (cost)</span><b>${a?a.cost+' vs '+b.cost:'—'}</b></div>
 <ul class="ev">${S.events.slice(-4).reverse().map(x=>`<li>h${x.hours}: ${esc(x.message||x.reason)}</li>`).join('')}</ul>
 <div class="ctl"><button class="btn g" id="nx" ${done?'disabled':''}>Next hour</button><button class="btn g" id="rs">Reset</button></div>`;
 $('#nx').onclick=()=>{stop();step()};$('#rs').onclick=()=>{stop();reset()};
}
const render=()=>{drawMap();panels();extras()};
async function load(){S=(await api('simulation/start',{scenario_id:SID})).initial_state;render();lastChanges=S.route_changes.length}
async function reset(){const d=await api('simulation/start',{scenario_id:SID,restart:true});S=d.initial_state;lastChanges=0;render();say('bot','Simulation reset to hour 0.')}
async function step(){
 try{const d=await api('simulation/step',{scenario_id:SID});S=d.state;render();
  if(S.route_changes.length>lastChanges){lastChanges=S.route_changes.length;say('bot','<b>Route changed.</b> '+esc(S.route_changes.at(-1).reason))}
  if(['DONE','FAILED'].includes(S.status)){stop();say('bot',S.status==='DONE'?`Mission complete: ${S.metrics.victims_rescued} of ${S.metrics.victims_total} people rescued.`:'Mission failed: '+esc(S.agents[0].fail_reason||'no safe route'))}
 }catch(e){stop();say('bot','Could not advance: '+esc(e.message))}
}
function play(){if(timer)return;timer=setInterval(step,1800);step();panels()}
function stop(){clearInterval(timer);timer=null;panels()}

/* ---------- chat ---------- */
function say(who,html){const d=document.createElement('div');d.className='m '+who;d.innerHTML=html;log.append(d);log.scrollTop=1e9;return d}
function confirmBox(el,yes){const a=document.createElement('div');a.className='acts';a.innerHTML='<button class="btn">Yes, start</button><button class="btn g">Cancel</button>';
 a.children[0].onclick=()=>{a.remove();yes()};a.children[1].onclick=()=>{a.remove();say('bot','Okay, nothing was started.')};el.append(a)}
function local(t){
 const H=S.hazards,e=H.exposure,m=S.metrics,rt=S.routes.find(r=>r.status==='OPEN'),x=t.toLowerCase();
 if(/plan|mission|evacu|rescue/.test(x)){const el=say('bot',`I've prepared a rescue plan: <b>${m.victims_remaining}</b> people to rescue, ${S.agents.length} robot, route cost ${rt?rt.cost:'—'} (risk-aware A*, victim order by CSP). Start the simulation?`);confirmBox(el,play)}
 else if(/why|chang/.test(x)){const c=S.route_changes.at(-1);say('bot',c?esc(c.reason):'The route has not changed yet. It re-plans automatically when the flood blocks a cell on the way.')}
 else if(/affect|people|how many/.test(x))say('bot',`About <b>${e.estimated_people_at_risk}</b> people are at risk at hour ${H.hours}; ${e.blocked_cells} cells are flooded. ${esc(S.disclaimer)}`);
 else if(/safe|route/.test(x))say('bot',rt?`The safest open route runs ${rt.path.length-1} cells to ${rt.target_type==='victim'?'the next person':'the safe zone'} (cost ${rt.cost}). It is drawn in green on the map.`:'No open route right now.');
 else if(/risk|flood|map/.test(x)){$('#view').value='live';render();say('bot',`Flood risk at hour ${H.hours}: ${e.blocked_cells} flooded cells, ${e.low_risk_cells+e.medium_risk_cells} at elevated risk, ~${e.estimated_people_at_risk} people exposed. The risk overlay is on the map.`)}
 else say('bot','I can explain flood risk, the safest route, how many people are affected, or plan a rescue. Try one of the buttons below.')}
async function ask(t,confirm=false){
 if(!confirm)say('me',esc(t));
 try{const r=await api('ai/ask',{message:t+' (scenario_id: '+SID+')',session_id:sess,confirm});aiOK=true;ast.textContent='Online';ast.className='';
  const el=say('bot',esc(r.reply||r.message||'Done.'));
  if(r.status==='CONFIRMATION_REQUIRED')confirmBox(el,async()=>{await ask('Yes, start',true);S=await api('simulation/state?scenario_id='+SID);render();play()});
 }catch(e){aiOK=false;ast.textContent='Offline – using built-in answers';ast.className='off';local(t)}}
f.onsubmit=e=>{e.preventDefault();const t=q.value.trim();if(t){q.value='';ask(t)}};
['Show flood risk map','Find safest route','How many people are affected?','Plan evacuation'].forEach(t=>{const b=document.createElement('button');b.textContent=t;b.onclick=()=>ask(t);chips.append(b)});

/* ---------- shell ---------- */
document.querySelectorAll('nav [data-go]').forEach(b=>b.onclick=()=>{
 document.querySelectorAll('nav .on').forEach(n=>n.classList.remove('on'));b.classList.add('on');const g=b.dataset.go;
 if(g==='safe'){map.flyTo(ll(...S.exit.position),18);return}
 if(g==='set'){dlgb.innerHTML=`<h3>Settings</h3><p>Scenario: <b>${SID}</b><br>Grid: ${area.grid_width}×${area.grid_height}, ${area.cell_size_meters} m per cell</p><button class="btn g" id="rr">Reset simulation</button>`;rr.onclick=()=>{dlg.close();stop();reset()};dlg.showModal();return}
 const el=document.getElementById(g);el.scrollIntoView({behavior:'smooth',block:'nearest'});el.classList.add('flash');setTimeout(()=>el.classList.remove('flash'),1000);if(g==='chat')q.focus()});
$('#theme').onclick=()=>{const d=document.documentElement;d.dataset.theme=d.dataset.theme==='dark'?'':'dark'};
$('#view').onchange=render;
document.querySelectorAll('[name=t]').forEach(r=>r.onchange=()=>{Object.values(tiles).forEach(t=>map.removeLayer(t));tiles[r.value].addTo(map)});
(async()=>{
 try{area=await api('geo/area');snaps=(await api('hazards/simulate',{scenario_id:SID})).snapshots;loc.textContent=area.name.replace(' (illustrative demo area)','');
  map=L.map('lmap').fitBounds([[area.bounds.min_lat,area.bounds.min_lon],[area.bounds.max_lat,area.bounds.max_lon]],{padding:[30,30]});
  tiles={sat:L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:18,attribution:'Esri'}),
   str:L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap'}),
   ter:L.tileLayer('https://tile.opentopomap.org/{z}/{x}/{y}.png',{maxZoom:17,attribution:'© OpenTopoMap'})};
  tiles.sat.addTo(map);lay=L.layerGroup().addTo(map);await load();ast.textContent='Online';
  say('bot','Hi! I\'m RescueBot. Ask about flood risk, the safest route, or plan a rescue mission.');
 }catch(e){say('bot','Could not reach the backend ('+esc(e.message)+'). Start it with <code>python api.py</code> (port 5000) and reload.')}
})();

/* ---------- pro enhancements ---------- */
let speed=1,prevK={};
document.body.insertAdjacentHTML('beforeend','<div class="toasts" id="toasts"></div><canvas id="cf"></canvas>');
function toast(m,t=''){const d=document.createElement('div');d.className='toast '+t;d.textContent=m;toasts.append(d);setTimeout(()=>d.remove(),4500)}
function extras(){const H=haz(),m=S.metrics;
 const k=[['People at risk',H.exposure.estimated_people_at_risk,'#e5484d'],['Rescued',m.victims_rescued+' / '+m.victims_total,'#16a34a'],['Flooded cells',H.exposure.blocked_cells,'#2563eb'],['Route changes',m.route_changes,'#f5a524'],['Flood hour',S.hours+' / 10','#7c3aed']];
 kpis.innerHTML=k.map(([l,v,c])=>`<div class="kp ${prevK[l]!==undefined&&prevK[l]!==v?'pop':''}" style="--c:${c}"><small>${l}</small><b>${v}</b></div>`).join('');k.forEach(([l,v])=>prevK[l]=v)}
const _say=say;say=function(w,h){const d=_say(w,h);
 if(w==='bot'&&/Route changed|Mission complete|Mission failed/.test(h))toast(h.replace(/<[^>]+>/g,''),/failed/.test(h)?'bad':/complete/.test(h)?'good':'warn');
 if(/Mission complete/.test(h))confetti();return d};
const _api=api;api=async function(p,b){let t;if(p==='ai/ask')t=_say('bot','<span class="dots"><i></i><i></i><i></i></span>');try{return await _api(p,b)}finally{t&&t.remove()}};
play=function(){if(timer)return;timer=setInterval(step,1800/speed);step();panels()};
$('#spd').onchange=e=>{speed=+e.target.value;if(timer){clearInterval(timer);timer=setInterval(step,1800/speed)}};
$('#demo').onclick=async()=>{stop();await reset();$('#view').value='live';$('#spd').value='2';speed=2;toast('Live demo started – watch the robot re-plan as the flood rises','good');play()};
$('#rep').onclick=()=>{
 const m=S.metrics,H=haz(),xs=snaps.map(s=>s.hours),W=520,HT=230,P=38,n=Math.max(1,xs.length-1);
 const X=i=>P+i*(W-2*P)/n;
 const chart=(t,vals,col)=>{const mx=Math.max(1,...vals),Y=v=>HT-P-v*(HT-2*P)/mx;
  return `<div class="c"><h3>${t}</h3><svg viewBox="0 0 ${W} ${HT}"><line x1="${P}" y1="${HT-P}" x2="${W-P}" y2="${HT-P}" stroke="#94a3b8"/>`+
  xs.map((h,i)=>`<text x="${X(i)}" y="${HT-14}" font-size="11" text-anchor="middle" fill="#64748b">h${h}</text>`).join('')+
  `<polyline fill="none" stroke="${col}" stroke-width="3" points="${vals.map((v,i)=>X(i)+','+Y(v)).join(' ')}"/>`+
  vals.map((v,i)=>`<circle cx="${X(i)}" cy="${Y(v)}" r="4.5" fill="${col}"/><text x="${X(i)}" y="${Y(v)-9}" font-size="11" font-weight="700" text-anchor="middle">${v}</text>`).join('')+`</svg></div>`};
 const pct=m.victims_total?Math.round(100*m.victims_rescued/m.victims_total):0;
 const cells=H.risk_grid.flat().map(v=>`<i style="background:${COL[v]||'#16a34a55'}"></i>`).join('');
 const html=`<!doctype html><meta charset="utf-8"><title>RescueBot Mission Report</title><style>
 body{font:15px/1.5 Inter,system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 20px;color:#14223a}
 h1{margin:0}h2{margin:26px 0 8px;border-bottom:2px solid #e2e8f0;padding-bottom:4px}.sub{color:#64748b}
 .k{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}.k div{border:1px solid #e2e8f0;border-left:4px solid #2563eb;border-radius:10px;padding:10px}
 .k small{display:block;color:#64748b;font-size:11px;text-transform:uppercase}.k b{font-size:22px}
 .ch{display:grid;grid-template-columns:1fr 1fr;gap:14px}.c{border:1px solid #e2e8f0;border-radius:10px;padding:10px}.c h3{margin:0 0 4px;font-size:14px}svg{width:100%}
 .bar{height:18px;background:#e2e8f0;border-radius:9px;overflow:hidden}.bar i{display:block;height:100%;background:#16a34a}
 .g{display:grid;grid-template-columns:repeat(${area.grid_width},1fr);gap:2px;max-width:360px}.g i{aspect-ratio:1;border-radius:3px}
 button{padding:8px 16px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer}@media print{button{display:none}}</style>
 <button onclick="print()">Save as PDF / Print</button>
 <h1>RescueBot – Mission Report</h1><div class="sub">${esc(area.name)} · ${new Date().toLocaleString()} · Status: ${S.status}</div>
 <div class="k"><div><small>Rescued</small><b>${m.victims_rescued} / ${m.victims_total}</b></div><div><small>Flood hour</small><b>${S.hours} / 10</b></div>
 <div><small>Route changes</small><b>${m.route_changes}</b></div><div><small>People at risk</small><b>${H.exposure.estimated_people_at_risk}</b></div></div>
 <div class="sub">Rescue progress ${pct}%</div><div class="bar"><i style="width:${pct}%"></i></div>
 <h2>Flood progression</h2><div class="ch">${chart('People at risk by hour',snaps.map(s=>s.exposure.estimated_people_at_risk),'#e5484d')}${chart('Flooded (blocked) cells by hour',snaps.map(s=>s.exposure.blocked_cells),'#2563eb')}</div>
 <h2>Risk map at hour ${H.hours}</h2><div class="g">${cells}</div><div class="sub">Green safe · Orange medium · Red high · Blue flooded</div>
 <h2>Event log</h2><ul>${S.events.map(x=>`<li>h${x.hours}: ${esc(x.message||x.reason)}</li>`).join('')}</ul>
 <p class="sub">${esc(S.disclaimer)}</p>`;
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([html],{type:'text/html'}));a.download='rescuebot-report.html';a.click();toast('Report with charts downloaded','good')};
addEventListener('keydown',e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||!S)return;const k=e.key.toLowerCase();
 if(k===' '){e.preventDefault();timer?stop():play()}else if(k==='n'){stop();step()}else if(k==='r'){stop();reset()}else if(k==='d')$('#theme').click()});
const mw=setInterval(()=>{if(!map)return;clearInterval(mw);map.on('click',e=>{
 const r=Math.floor((area.bounds.max_lat-e.latlng.lat)/dLat()),c=Math.floor((e.latlng.lng-area.bounds.min_lon)/dLon()),v=haz().risk_grid[r]?.[c];if(v===undefined)return;
 L.popup().setLatLng(e.latlng).setContent(`<b>Cell (${r}, ${c})</b><br>${['Safe','Medium risk','High risk','Flooded – blocked'][v]}`).openOn(map)})},300);
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
if(SR){const b=document.createElement('button');b.type='button';b.className='btn g mic';b.title='Voice input';b.textContent='🎤';f.insertBefore(b,f.lastElementChild);
 b.onclick=()=>{const r=new SR();r.lang='en-US';b.classList.add('on');r.onresult=e=>{q.value=e.results[0][0].transcript;f.requestSubmit()};r.onend=()=>b.classList.remove('on');r.start()}}
function confetti(){const c=cf,x=c.getContext('2d');c.width=innerWidth;c.height=innerHeight;
 const P=Array.from({length:150},()=>({x:Math.random()*c.width,y:-20-Math.random()*200,v:2+Math.random()*4,s:4+Math.random()*6,h:Math.random()*360}));let n=0;
 (function fr(){x.clearRect(0,0,c.width,c.height);P.forEach(p=>{p.y+=p.v;p.x+=Math.sin(p.y/30);x.fillStyle=`hsl(${p.h},80%,55%)`;x.fillRect(p.x,p.y,p.s,p.s*.6)});if(++n<260)requestAnimationFrame(fr);else x.clearRect(0,0,c.width,c.height)})()}

/* ---------- Word report ---------- */
function png(w,h,fn){const c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d');x.fillStyle='#fff';x.fillRect(0,0,w,h);fn(x);return c.toDataURL('image/png').split(',')[1]}
function lineImg(t,labels,vals,col){return png(640,300,x=>{const W=640,H=300,P=50,mx=Math.max(1,...vals),n=Math.max(1,vals.length-1),X=i=>P+i*(W-2*P)/n,Y=v=>H-P-v*(H-2*P)/mx;
 x.fillStyle='#14223a';x.font='bold 16px Arial';x.fillText(t,P,26);x.strokeStyle='#94a3b8';x.beginPath();x.moveTo(P,40);x.lineTo(P,H-P);x.lineTo(W-P,H-P);x.stroke();
 x.strokeStyle=col;x.lineWidth=3;x.beginPath();vals.forEach((v,i)=>i?x.lineTo(X(i),Y(v)):x.moveTo(X(i),Y(v)));x.stroke();
 x.font='13px Arial';x.textAlign='center';vals.forEach((v,i)=>{x.fillStyle=col;x.beginPath();x.arc(X(i),Y(v),5,0,7);x.fill();x.fillStyle='#14223a';x.fillText(v,X(i),Y(v)-10);x.fillStyle='#64748b';x.fillText('h'+labels[i],X(i),H-P+20)})})}
function gridImg(g){const k=16,cl={0:'#86d9a5',1:'#f5a524',2:'#e5484d',3:'#2563eb'};return png(g[0].length*k,g.length*k,x=>g.forEach((row,r)=>row.forEach((v,c)=>{x.fillStyle=cl[v];x.fillRect(c*k,r*k,k-1,k-1)})))}

/*DOCX_START*/
const CRC=(()=>{const t=[];for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=c&1?0xEDB88320^(c>>>1):c>>>1;t[n]=c>>>0}return t})();
const crc32=u=>{let c=-1;for(let i=0;i<u.length;i++)c=CRC[(c^u[i])&255]^(c>>>8);return(c^-1)>>>0};
function zipFiles(files){const te=new TextEncoder(),parts=[],cd=[];let off=0;
 const u16=v=>[v&255,v>>8&255],u32=v=>[v&255,v>>8&255,v>>16&255,v>>>24&255];
 files.forEach(([name,data])=>{const n=te.encode(name),d=typeof data==='string'?te.encode(data):data,crc=crc32(d);
  const lh=new Uint8Array([0x50,0x4b,3,4,...u16(20),...u16(0),...u16(0),...u16(0),...u16(0x21),...u32(crc),...u32(d.length),...u32(d.length),...u16(n.length),...u16(0)]);
  parts.push(lh,n,d);
  cd.push(new Uint8Array([0x50,0x4b,1,2,...u16(20),...u16(20),...u16(0),...u16(0),...u16(0),...u16(0x21),...u32(crc),...u32(d.length),...u32(d.length),...u16(n.length),...u16(0),...u16(0),...u16(0),...u16(0),...u32(0),...u32(off)]),n);
  off+=lh.length+n.length+d.length});
 const cl=cd.reduce((a,x)=>a+x.length,0);
 const end=new Uint8Array([0x50,0x4b,5,6,0,0,0,0,...u16(files.length),...u16(files.length),...u32(cl),...u32(off),0,0]);
 return new Blob([...parts,...cd,end],{type:'application/vnd.openxmlformats-officedocument.wordprocessingml.document'})}
function buildDocx(title,sub,rows,imgs,heads,events,legend,disc){
 const run=(t,o={})=>`<w:r><w:rPr>${o.b?'<w:b/>':''}${o.c?`<w:color w:val="${o.c}"/>`:''}<w:sz w:val="${(o.s||11)*2}"/></w:rPr><w:t xml:space="preserve">${esc(t)}</w:t></w:r>`;
 const para=(t,o={})=>`<w:p><w:pPr><w:spacing w:before="${o.sb||60}" w:after="${o.sa||60}"/></w:pPr>${run(t,o)}</w:p>`;
 const head=t=>para(t,{b:1,s:14,c:'2563EB',sb:280});
 const cell=(t,b,w,fill)=>`<w:tc><w:tcPr><w:tcW w:w="${w}" w:type="dxa"/>${fill?`<w:shd w:val="clear" w:color="auto" w:fill="${fill}"/>`:''}</w:tcPr>${para(t,{b})}</w:tc>`;
 const table=`<w:tbl><w:tblPr><w:tblW w:w="9000" w:type="dxa"/><w:tblBorders>${['top','left','bottom','right','insideH','insideV'].map(s=>`<w:${s} w:val="single" w:sz="4" w:color="CBD5E1"/>`).join('')}</w:tblBorders></w:tblPr>${rows.map(([a,b])=>`<w:tr>${cell(a,1,3200,'EEF2F7')}${cell(b,0,5800)}</w:tr>`).join('')}</w:tbl>`;
 const pic=(i,w,h)=>{const cx=w*9525,cy=Math.round(h*9525);return `<w:p><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="${cx}" cy="${cy}"/><wp:docPr id="${i+1}" name="Image ${i+1}"/><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="${i+1}" name="img${i+1}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rId${i+1}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="${cx}" cy="${cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>`};
 const body=para(title,{b:1,s:24,c:'0F2747'})+para(sub,{c:'64748B'})+head('Summary')+table+head(heads[0])+pic(0,imgs[0][1],imgs[0][2])+pic(1,imgs[1][1],imgs[1][2])+head(heads[1])+pic(2,imgs[2][1],imgs[2][2])+para(legend,{s:9,c:'64748B'})+head('Event log')+events.map(e=>para('• '+e)).join('')+para(disc,{s:9,c:'64748B',sb:240});
 const doc=`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"><w:body>${body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>`;
 const H='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';
 const b2u=b=>Uint8Array.from(atob(b),c=>c.charCodeAt(0));
 return zipFiles([['[Content_Types].xml',H+'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'],
  ['_rels/.rels',H+`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${R}/officeDocument" Target="word/document.xml"/></Relationships>`],
  ['word/document.xml',doc],
  ['word/_rels/document.xml.rels',H+`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${imgs.map((_,i)=>`<Relationship Id="rId${i+1}" Type="${R}/image" Target="media/img${i+1}.png"/>`).join('')}</Relationships>`],
  ...imgs.map((x,i)=>[`word/media/img${i+1}.png`,b2u(x[0])])])}
/*DOCX_END*/
$('#wrd').onclick=()=>{
 const m=S.metrics,H=haz(),g=H.risk_grid,xs=snaps.map(s=>s.hours),pct=m.victims_total?Math.round(100*m.victims_rescued/m.victims_total):0,gw=g[0].length*16,gh=g.length*16;
 const blob=buildDocx('RescueBot – Mission Report',area.name+' · '+new Date().toLocaleString(),
  [['Status',S.status],['Flood hour',S.hours+' of 10'],['People rescued',m.victims_rescued+' / '+m.victims_total+' ('+pct+'%)'],['Route changes',m.route_changes],['People at risk (hour '+H.hours+')',H.exposure.estimated_people_at_risk],['Flooded cells',H.exposure.blocked_cells],['Method','CSP + Risk-aware A*']],
  [[lineImg('People at risk by hour',xs,snaps.map(s=>s.exposure.estimated_people_at_risk),'#e5484d'),560,262],[lineImg('Flooded (blocked) cells by hour',xs,snaps.map(s=>s.exposure.blocked_cells),'#2563eb'),560,262],[gridImg(g),360,360*gh/gw]],
  ['Flood progression','Risk map – hour '+H.hours],S.events.map(x=>'h'+x.hours+': '+(x.message||x.reason)),
  'Green: safe · Orange: medium risk · Red: high risk · Blue: flooded (blocked)',S.disclaimer);
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='rescuebot-report.docx';a.click();toast('Word report downloaded','good')};