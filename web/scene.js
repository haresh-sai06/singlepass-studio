/* ============================================================
   Singlepass Studio — viewer scene
   Lightweight 3D point-cloud renderer on raw canvas 2D.
   No dependencies: projection, depth sorting and confidence
   shading are all done here so the wireframe runs offline.
   ============================================================ */

const CONF = {
  3: [111, 169, 138],   // well triangulated  — sage
  2: [201, 162, 39],    // weak               — brass
  1: [199, 125, 74],    // monocular guess    — terracotta
  0: [180, 84, 74],     // never observed     — muted red
};

const MATERIAL = [237, 235, 231];

function rand(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/* ---- scene generation: terrain + blocks, mirrors probe/scene.py ---- */
function buildScene() {
  const r = rand(7);
  const pts = [];

  const terrainH = (x, y) =>
    2.4 * Math.sin(x / 45) * Math.cos(y / 55) + 1.2 * Math.sin(x / 18 + 1.3);

  // terrain
  for (let i = 0; i < 9000; i++) {
    const x = (r() - 0.5) * 200;
    const y = (r() - 0.5) * 200;
    const z = terrainH(x, y);
    // coverage falls off away from the flight corridor (y ≈ -35)
    const d = Math.abs(y + 35);
    let conf = d < 42 ? 3 : d < 62 ? 2 : d < 76 ? 1 : 0;
    if (Math.abs(x) > 88) conf = Math.min(conf, 1);
    pts.push({ x, y, z, conf, base: [56 + 26 * r(), 68 + 22 * r(), 42 + 18 * r()] });
  }

  // buildings
  const blocks = [];
  for (let b = 0; b < 9; b++) {
    const cx = (r() - 0.5) * 128;
    const cy = (r() - 0.5) * 118;
    const w = 8 + r() * 14, dp = 8 + r() * 14, h = 7 + r() * 23;
    const bz = terrainH(cx, cy);
    blocks.push({ cx, cy, w, dp, h, bz });

    const tone = 132 + r() * 38;
    // roof
    for (let i = 0; i < 620; i++) {
      pts.push({
        x: cx + (r() - 0.5) * w, y: cy + (r() - 0.5) * dp, z: bz + h,
        conf: 3, base: [tone * 0.82, tone * 0.8, tone * 0.76],
      });
    }
    // four façades — the side facing away from the flight line is unobserved
    const faces = [[1, 0], [-1, 0], [0, 1], [0, -1]];
    for (const [sx, sy] of faces) {
      const facingCamera = sy === -1 || (sy === 0 && cy > -35);
      const conf = facingCamera ? 3 : (sy === 1 ? 0 : 2);
      for (let i = 0; i < 340; i++) {
        const px = sx ? cx + sx * w / 2 : cx + (r() - 0.5) * w;
        const py = sx ? cy + (r() - 0.5) * dp : cy + sy * dp / 2;
        pts.push({ x: px, y: py, z: bz + r() * h, conf, base: [tone, tone * 0.97, tone * 0.93] });
      }
    }
  }
  return { pts, blocks };
}

/* ---- renderer ---- */
export function mountViewer(canvas, opts = {}) {
  const ctx = canvas.getContext('2d');
  const { pts, blocks } = buildScene();

  const state = {
    yaw: -0.62, pitch: 0.52, dist: 235,
    target: [0, -10, 8],
    mode: 'confidence',      // 'confidence' | 'material'
    autorotate: true,
    dragging: false, lx: 0, ly: 0,
  };

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, r.width * dpr);
    canvas.height = Math.max(1, r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  canvas.addEventListener('pointerdown', e => {
    state.dragging = true; state.autorotate = false;
    state.lx = e.clientX; state.ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', e => {
    state.dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
  });
  canvas.addEventListener('pointermove', e => {
    if (!state.dragging) return;
    state.yaw += (e.clientX - state.lx) * 0.006;
    state.pitch = Math.max(0.08, Math.min(1.45, state.pitch + (e.clientY - state.ly) * 0.005));
    state.lx = e.clientX; state.ly = e.clientY;
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    state.dist = Math.max(90, Math.min(460, state.dist * (1 + Math.sign(e.deltaY) * 0.09)));
  }, { passive: false });

  function frame() {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (state.autorotate) state.yaw += 0.0016;

    // camera
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const cy_ = Math.cos(state.yaw), sy_ = Math.sin(state.yaw);
    const eye = [
      state.target[0] + state.dist * cp * sy_,
      state.target[1] + state.dist * cp * cy_,
      state.target[2] + state.dist * sp,
    ];
    // basis
    const f = norm(sub(state.target, eye));
    const rgt = norm(cross(f, [0, 0, 1]));
    const up = cross(rgt, f);
    const fl = H * 1.15;

    // backdrop — subtle vertical wash
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, '#141417');
    g.addColorStop(1, '#0C0C0E');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    drawGrid(ctx, W, H, eye, rgt, up, f, fl);

    // project + depth sort
    const buf = [];
    for (const p of pts) {
      const d = [p.x - eye[0], p.y - eye[1], p.z - eye[2]];
      const z = dot(d, f);
      if (z < 6) continue;
      const sx = W / 2 + (dot(d, rgt) / z) * fl;
      const sy = H / 2 - (dot(d, up) / z) * fl;
      if (sx < -20 || sx > W + 20 || sy < -20 || sy > H + 20) continue;
      buf.push({ sx, sy, z, p });
    }
    buf.sort((a, b) => b.z - a.z);

    for (const b of buf) {
      const col = state.mode === 'confidence' ? CONF[b.p.conf] : b.p.base;
      const fade = Math.max(0.18, Math.min(1, 1 - (b.z - 110) / 340));
      const size = Math.max(0.9, 190 / b.z);
      ctx.globalAlpha = fade * (state.mode === 'confidence' && b.p.conf === 0 ? 0.85 : 0.95);
      ctx.fillStyle = `rgb(${col[0] | 0},${col[1] | 0},${col[2] | 0})`;
      ctx.fillRect(b.sx, b.sy, size, size);
    }
    ctx.globalAlpha = 1;

    drawFlightPath(ctx, W, H, eye, rgt, up, f, fl);
    if (opts.onFrame) opts.onFrame(state);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return {
    setMode: m => { state.mode = m; },
    toggleRotate: () => (state.autorotate = !state.autorotate),
    reset: () => { state.yaw = -0.62; state.pitch = 0.52; state.dist = 235; state.autorotate = true; },
  };

  /* -- helpers -- */
  function project(pw, eye, rgt, up, f, fl, W, H) {
    const d = sub(pw, eye);
    const z = dot(d, f);
    if (z < 4) return null;
    return [W / 2 + (dot(d, rgt) / z) * fl, H / 2 - (dot(d, up) / z) * fl, z];
  }

  function drawGrid(ctx, W, H, eye, rgt, up, f, fl) {
    ctx.strokeStyle = 'rgba(255,255,255,0.030)';
    ctx.lineWidth = 1;
    const S = 100, STEP = 20;
    for (let i = -S; i <= S; i += STEP) {
      strokeLine([[i, -S, -3], [i, S, -3]]);
      strokeLine([[-S, i, -3], [S, i, -3]]);
    }
    function strokeLine(seg) {
      const a = project(seg[0], eye, rgt, up, f, fl, W, H);
      const b = project(seg[1], eye, rgt, up, f, fl, W, H);
      if (!a || !b) return;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
  }

  function drawFlightPath(ctx, W, H, eye, rgt, up, f, fl) {
    const pts2 = [];
    for (let i = 0; i <= 40; i++) {
      const x = -80 + (160 * i) / 40;
      const q = project([x, -35, 90], eye, rgt, up, f, fl, W, H);
      if (q) pts2.push(q);
    }
    if (pts2.length < 2) return;
    ctx.strokeStyle = 'rgba(201,162,39,0.55)';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    pts2.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
    ctx.stroke();
    ctx.setLineDash([]);
    // camera stations
    ctx.fillStyle = 'rgba(227,197,103,0.9)';
    for (let i = 0; i < pts2.length; i += 5) {
      ctx.fillRect(pts2[i][0] - 1.6, pts2[i][1] - 1.6, 3.2, 3.2);
    }
  }
}

/* vec helpers */
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function norm(a) {
  const m = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / m, a[1] / m, a[2] / m];
}
