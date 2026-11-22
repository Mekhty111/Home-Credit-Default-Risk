import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { BANDS, cssVar, getTheme, reducedMotion } from './core.js';

const CARD_W = 3.4;
const CARD_H = CARD_W / 1.586;
const CARD_R = 0.2;
const CARD_D = 0.035;
const BEVEL = 0.018;

const easeOutCubic = (x) => 1 - Math.pow(1 - x, 3);
const easeInOut = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const lerp = (a, b, k) => a + (b - a) * k;

function roundedRect(w, h, r) {
  const s = new THREE.Shape();
  const x = -w / 2, y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.absarc(x + w - r, y + r, r, -Math.PI / 2, 0, false);
  s.lineTo(x + w, y + h - r);
  s.absarc(x + w - r, y + h - r, r, 0, Math.PI / 2, false);
  s.lineTo(x + r, y + h);
  s.absarc(x + r, y + h - r, r, Math.PI / 2, Math.PI, false);
  s.lineTo(x, y + r);
  s.absarc(x + r, y + r, r, Math.PI, Math.PI * 1.5, false);
  return s;
}

function planarUV(geometry, w, h, flipX = false) {
  const pos = geometry.attributes.position;
  const uv = new Float32Array(pos.count * 2);
  for (let i = 0; i < pos.count; i++) {
    const u = (pos.getX(i) + w / 2) / w;
    uv[i * 2] = flipX ? 1 - u : u;
    uv[i * 2 + 1] = (pos.getY(i) + h / 2) / h;
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return geometry;
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

function drawFront(score = 583, bandLabel = 'Very low', bandColor = '#33c48a') {
  const W = 1600, H = Math.round(W / 1.586);
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');

  const bg = g.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, '#1d2658');
  bg.addColorStop(0.45, '#10132a');
  bg.addColorStop(1, '#07080f');
  g.fillStyle = bg; g.fillRect(0, 0, W, H);

  let rg = g.createRadialGradient(W * 0.12, H * 0.05, 0, W * 0.12, H * 0.05, W * 0.75);
  rg.addColorStop(0, 'rgba(95,150,255,.55)'); rg.addColorStop(1, 'rgba(95,150,255,0)');
  g.fillStyle = rg; g.fillRect(0, 0, W, H);
  rg = g.createRadialGradient(W * 0.95, H * 1.05, 0, W * 0.95, H * 1.05, W * 0.6);
  rg.addColorStop(0, 'rgba(233,138,90,.38)'); rg.addColorStop(1, 'rgba(233,138,90,0)');
  g.fillStyle = rg; g.fillRect(0, 0, W, H);

  // Guilloche
  g.save();
  g.strokeStyle = 'rgba(255,255,255,.045)';
  g.lineWidth = 2;
  for (let i = 0; i < 46; i++) {
    g.beginPath();
    g.ellipse(W * 0.82, H * 0.38, 120 + i * 26, 60 + i * 15, -0.35, 0, Math.PI * 2);
    g.stroke();
  }
  g.restore();

  // Brand mark
  const bx = 96, by = 86, bs = 96;
  const bgm = g.createLinearGradient(bx, by, bx + bs, by + bs);
  bgm.addColorStop(0, '#3346e0'); bgm.addColorStop(1, '#4f8bff');
  g.fillStyle = bgm; roundRectPath(g, bx, by, bs, bs, 28); g.fill();
  g.strokeStyle = '#fff'; g.lineWidth = 8; g.lineCap = 'round';
  g.beginPath(); g.arc(bx + bs / 2, by + bs / 2, 24, Math.PI * 0.2, Math.PI * 1.8, false); g.stroke();
  g.fillStyle = '#fff'; g.beginPath(); g.arc(bx + bs / 2 + 21, by + bs / 2, 6.5, 0, Math.PI * 2); g.fill();
  g.font = '600 66px Geist, system-ui, sans-serif';
  g.fillStyle = '#fff'; g.textBaseline = 'middle';
  g.fillText('Credence', bx + bs + 30, by + bs / 2 + 2);

  // Contactless
  g.strokeStyle = 'rgba(255,255,255,.75)'; g.lineWidth = 7;
  for (let i = 0; i < 4; i++) {
    g.beginPath(); g.arc(W - 170, by + bs / 2, 16 + i * 17, -Math.PI / 3.4, Math.PI / 3.4); g.stroke();
  }

  // Chip
  const cx = 110, cy = 330, cw = 200, ch = 150;
  const chip = g.createLinearGradient(cx, cy, cx + cw, cy + ch);
  chip.addColorStop(0, '#f6dc96'); chip.addColorStop(0.5, '#b98d34'); chip.addColorStop(1, '#f1d282');
  g.fillStyle = chip; roundRectPath(g, cx, cy, cw, ch, 26); g.fill();
  g.strokeStyle = 'rgba(80,55,10,.45)'; g.lineWidth = 3;
  g.beginPath();
  g.moveTo(cx, cy + ch / 2); g.lineTo(cx + cw * 0.36, cy + ch / 2);
  g.moveTo(cx + cw * 0.64, cy + ch / 2); g.lineTo(cx + cw, cy + ch / 2);
  g.moveTo(cx + cw * 0.36, cy); g.lineTo(cx + cw * 0.36, cy + ch);
  g.moveTo(cx + cw * 0.64, cy); g.lineTo(cx + cw * 0.64, cy + ch);
  g.stroke();
  roundRectPath(g, cx + cw * 0.36, cy + ch * 0.28, cw * 0.28, ch * 0.44, 10); g.stroke();

  // Number
  g.font = '500 82px "Geist Mono", ui-monospace, monospace';
  g.fillStyle = 'rgba(255,255,255,.92)';
  g.textBaseline = 'alphabetic';
  if ('letterSpacing' in g) g.letterSpacing = '6px';
  g.fillText('5412  7834  ••••  0583', 110, 650);
  if ('letterSpacing' in g) g.letterSpacing = '3px';

  // Footer: score + band
  g.font = '500 30px Geist, system-ui, sans-serif';
  g.fillStyle = 'rgba(255,255,255,.5)';
  g.fillText('SCORECARD', 110, 790);
  g.fillText('RISK BAND', W - 470, 790);
  if ('letterSpacing' in g) g.letterSpacing = '0px';
  g.font = 'italic 400 104px "Instrument Serif", Georgia, serif';
  g.fillStyle = '#fff';
  g.fillText(String(score), 106, 892);
  const sw = g.measureText(String(score)).width;
  g.font = '500 36px Geist, system-ui, sans-serif';
  g.fillStyle = 'rgba(255,255,255,.55)';
  g.fillText('pts', 106 + sw + 16, 890);
  g.fillStyle = bandColor;
  g.beginPath(); g.arc(W - 452, 864, 13, 0, Math.PI * 2); g.fill();
  g.font = '500 54px Geist, system-ui, sans-serif';
  g.fillStyle = '#fff';
  g.fillText(bandLabel, W - 422, 884);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

function drawBack() {
  const W = 1024, H = Math.round(W / 1.586);
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  const bg = g.createLinearGradient(0, 0, W, H);
  bg.addColorStop(0, '#141833'); bg.addColorStop(1, '#07080f');
  g.fillStyle = bg; g.fillRect(0, 0, W, H);
  g.fillStyle = '#05060a'; g.fillRect(0, 70, W, 120);
  g.fillStyle = 'rgba(255,255,255,.85)'; g.fillRect(70, 250, 560, 80);
  g.fillStyle = 'rgba(0,0,0,.5)'; g.font = 'italic 400 44px "Instrument Serif", Georgia, serif';
  g.fillText('Credence Risk Engine', 96, 303);
  g.fillStyle = 'rgba(255,255,255,.4)'; g.font = '400 22px Geist, system-ui, sans-serif';
  g.fillText('WoE scorecard · PDO 20 · 600 @ 19:1 · LightGBM benchmark', 70, 420);
  g.fillText('Decisions are explainable by design.', 70, 456);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function dotTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const rg = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  rg.addColorStop(0, 'rgba(255,255,255,1)');
  rg.addColorStop(0.4, 'rgba(255,255,255,.6)');
  rg.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = rg; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

async function fontsReady() {
  try {
    await Promise.race([
      Promise.all([
        document.fonts.load('600 66px Geist'),
        document.fonts.load('500 82px "Geist Mono"'),
        document.fonts.load('italic 400 104px "Instrument Serif"'),
      ]),
      new Promise((r) => setTimeout(r, 1500)),
    ]);
  } catch {}
}

export async function mountHero(container, { score = 583, bandKey = 'very_low', bandLabel = 'Very low' } = {}) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  } catch {
    return null; // CSS fallback stays visible
  }
  await fontsReady();

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
  camera.position.set(0, 0, 10.5);

  scene.add(new THREE.AmbientLight(0xffffff, 0.35));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(3, 4, 5);
  scene.add(key);
  const rim = new THREE.PointLight(0x4f8bff, 18, 12);
  rim.position.set(-3, 1.5, -2);
  scene.add(rim);

  const root = new THREE.Group();
  scene.add(root);

  // ── Card
  const shape = roundedRect(CARD_W, CARD_H, CARD_R);
  const bodyGeo = new THREE.ExtrudeGeometry(shape, {
    depth: CARD_D, bevelEnabled: true, bevelThickness: BEVEL, bevelSize: BEVEL, bevelSegments: 5, curveSegments: 24,
  });
  bodyGeo.translate(0, 0, -CARD_D / 2);
  const faceGeo = planarUV(new THREE.ShapeGeometry(shape, 24), CARD_W, CARD_H);
  const backGeo = planarUV(new THREE.ShapeGeometry(shape, 24), CARD_W, CARD_H, true);

  const edgeMat = new THREE.MeshPhysicalMaterial({ color: 0x1a1f3d, metalness: 0.75, roughness: 0.28, clearcoat: 1, clearcoatRoughness: 0.12 });
  const frontTex = drawFront(score, bandLabel, cssVar(BANDS.find((b) => b.key === bandKey)?.color || '--band-vl'));
  const frontMat = new THREE.MeshPhysicalMaterial({
    map: frontTex, metalness: 0.25, roughness: 0.38, clearcoat: 1, clearcoatRoughness: 0.08,
    iridescence: 0.45, iridescenceIOR: 1.35, iridescenceThicknessRange: [120, 420],
  });
  const backMat = new THREE.MeshPhysicalMaterial({ map: drawBack(), metalness: 0.2, roughness: 0.45, clearcoat: 0.8 });

  const card = new THREE.Group();
  card.add(new THREE.Mesh(bodyGeo, edgeMat));
  const front = new THREE.Mesh(faceGeo, frontMat);
  front.position.z = CARD_D / 2 + BEVEL + 0.0015;
  card.add(front);
  const back = new THREE.Mesh(backGeo, backMat);
  back.position.z = -(CARD_D / 2 + BEVEL + 0.0015);
  back.rotation.y = Math.PI;
  card.add(back);
  root.add(card);

  // ── Glass card behind
  const glass = new THREE.Mesh(bodyGeo, new THREE.MeshPhysicalMaterial({
    color: 0xdfe6ff, metalness: 0, roughness: 0.18, transmission: 1, thickness: 0.5, ior: 1.45,
    clearcoat: 1, clearcoatRoughness: 0.1, transparent: true, opacity: 0.9,
  }));
  glass.position.set(-0.9, 0.75, -1.3);
  glass.rotation.set(0.15, 0.5, 0.22);
  glass.scale.setScalar(0.92);
  root.add(glass);

  // ── Risk ring: arc length proportional to each band's population share
  const ring = new THREE.Group();
  ring.rotation.set(1.18, -0.2, 0);
  const RING_R = 2.65;
  const gap = 0.07;
  const total = Math.PI * 2 - gap * BANDS.length;
  const shareSum = BANDS.reduce((s, b) => s + b.share, 0);
  const ringMats = [];
  let start = 0;
  for (const b of BANDS) {
    const len = (b.share / shareSum) * total;
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.15, emissiveIntensity: 0.45 });
    ringMats.push([mat, b.color]);
    const seg = new THREE.Mesh(new THREE.TorusGeometry(RING_R, 0.032, 12, 120, len), mat);
    seg.rotation.z = start;
    ring.add(seg);
    start += len + gap;
  }
  const bead = new THREE.Mesh(new THREE.SphereGeometry(0.075, 24, 24), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.9 }));
  ring.add(bead);
  const ring2 = new THREE.Mesh(new THREE.TorusGeometry(RING_R + 0.38, 0.006, 6, 200), new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.35 }));
  ring.add(ring2);
  root.add(ring);

  // ── Particles
  const N = 260;
  const pts = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r = 3 + Math.random() * 3.2;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(2 * Math.random() - 1);
    pts[i * 3] = r * Math.sin(ph) * Math.cos(th);
    pts[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th) * 0.7;
    pts[i * 3 + 2] = r * Math.cos(ph) - 1.5;
  }
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  const pMat = new THREE.PointsMaterial({ size: 0.05, map: dotTexture(), transparent: true, depthWrite: false, sizeAttenuation: true });
  const particles = new THREE.Points(pGeo, pMat);
  root.add(particles);

  // ── Theme
  const applyTheme = () => {
    const dark = getTheme() === 'dark';
    ringMats.forEach(([mat, v]) => {
      const col = new THREE.Color(cssVar(v));
      mat.color.copy(col);
      mat.emissive.copy(col);
    });
    pMat.color.set(dark ? 0xc9d4ff : 0x3346e0);
    pMat.opacity = dark ? 0.55 : 0.35;
    ring2.material.color.set(dark ? 0xffffff : 0x12141c);
    ring2.material.opacity = dark ? 0.16 : 0.12;
    rim.intensity = dark ? 26 : 14;
    renderer.toneMappingExposure = dark ? 1.0 : 1.08;
  };
  applyTheme();
  document.addEventListener('themechange', applyTheme);

  // ── Interaction
  const pointer = { x: 0, y: 0, tx: 0, ty: 0 };
  const onMove = (e) => {
    const r = container.getBoundingClientRect();
    pointer.tx = ((e.clientX - r.left) / r.width - 0.5) * 2;
    pointer.ty = ((e.clientY - r.top) / r.height - 0.5) * 2;
  };
  window.addEventListener('pointermove', onMove, { passive: true });

  let spin = null;
  const raycaster = new THREE.Raycaster();
  container.addEventListener('click', (e) => {
    const r = container.getBoundingClientRect();
    const v = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(v, camera);
    if (raycaster.intersectObject(card, true).length && !spin) spin = { t0: performance.now(), from: 0 };
  });
  container.addEventListener('pointermove', (e) => {
    const r = container.getBoundingClientRect();
    const v = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(v, camera);
    container.style.cursor = raycaster.intersectObject(card, true).length ? 'pointer' : '';
  });

  // ── Sizing
  const resize = () => {
    const { width, height } = container.getBoundingClientRect();
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    // Keep the composition framed on narrow screens
    const narrow = width < 560 ? 1.32 : 1;
    camera.position.z = (width / height < 1 ? 10.5 / Math.max(0.62, width / height) : 10.5) * narrow;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  resize();

  // ── Loop
  const clock = new THREE.Clock();
  let running = true, visible = true, elapsed = 0;
  const introStart = performance.now();

  const frame = () => {
    if (!running) return;
    const dt = Math.min(clock.getDelta(), 0.05);
    elapsed += dt;
    const intro = reducedMotion ? 1 : easeOutCubic(Math.min(1, (performance.now() - introStart) / 1900));

    pointer.x = lerp(pointer.x, pointer.tx, 0.05);
    pointer.y = lerp(pointer.y, pointer.ty, 0.05);

    root.rotation.y = pointer.x * 0.32;
    root.rotation.x = pointer.y * 0.2;

    const float = reducedMotion ? 0 : Math.sin(elapsed * 0.9) * 0.08;
    card.position.set(0.15, float + (1 - intro) * -0.9, (1 - intro) * -2.5);
    let spinY = 0;
    if (spin) {
      const p = Math.min(1, (performance.now() - spin.t0) / 1400);
      spinY = easeInOut(p) * Math.PI * 2;
      if (p >= 1) spin = null;
    }
    card.rotation.set(
      -0.16 + Math.sin(elapsed * 0.6) * 0.03,
      -0.42 + (1 - intro) * 1.6 + spinY,
      0.08 + Math.sin(elapsed * 0.5) * 0.02,
    );

    glass.position.y = 0.75 + Math.sin(elapsed * 0.7 + 1.2) * 0.1;
    glass.rotation.y = 0.5 + pointer.x * 0.12;
    glass.material.opacity = 0.9 * intro;

    ring.scale.setScalar(0.7 + 0.3 * intro);
    ring.rotation.z += reducedMotion ? 0 : dt * 0.06;
    const a = elapsed * 0.45;
    bead.position.set(Math.cos(a) * RING_R, Math.sin(a) * RING_R, 0);

    particles.rotation.y += reducedMotion ? 0 : dt * 0.025;
    pMat.opacity = (getTheme() === 'dark' ? 0.55 : 0.35) * intro;

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  };

  const setRunning = (next) => {
    if (next === running) return;
    running = next;
    if (running) { clock.getDelta(); requestAnimationFrame(frame); }
  };
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; setRunning(visible && !document.hidden); }).observe(container);
  document.addEventListener('visibilitychange', () => setRunning(visible && !document.hidden));

  requestAnimationFrame(frame);
  requestAnimationFrame(() => container.classList.add('is-ready'));

  return {
    setCard({ score: s, bandKey: k, bandLabel: l }) {
      frontMat.map.dispose();
      frontMat.map = drawFront(s, l, cssVar(BANDS.find((b) => b.key === k)?.color || '--band-vl'));
      frontMat.needsUpdate = true;
    },
  };
}
