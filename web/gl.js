/* ============================================================
   WebGL point-cloud renderer.

   Canvas 2D sorts and fills rects per point on the CPU — fine for 20 k,
   hopeless past 100 k. This uploads once to the GPU and draws the whole
   cloud in a single call, so 150 k points hold 60 fps.

   Two colour modes share one buffer set: true photographic colour sampled
   from the source frames, and triangulation-quality grading.
   ============================================================ */

const VERT = `
attribute vec3 aPos;
attribute vec3 aCol;
attribute float aGrade;
uniform mat4 uMVP;
uniform float uPointScale;
uniform float uMode;        // 0 = photo colour, 1 = quality grade
uniform float uMinGrade;    // hide points below this grade
varying vec3 vCol;
varying float vDrop;

vec3 gradeColour(float g){
  if(g > 2.5) return vec3(0.435, 0.663, 0.541);   // >=10 deg  sage
  if(g > 1.5) return vec3(0.788, 0.635, 0.153);   // 5-10      brass
  if(g > 0.5) return vec3(0.780, 0.490, 0.290);   // 2-5       terracotta
  return              vec3(0.706, 0.329, 0.290);  // <2        clay
}

void main(){
  vec4 clip = uMVP * vec4(aPos, 1.0);
  gl_Position = clip;
  float d = max(clip.w, 0.001);
  gl_PointSize = clamp(uPointScale / d, 1.0, 9.0);
  vCol = (uMode > 0.5) ? gradeColour(aGrade) : aCol;
  vDrop = (aGrade + 0.01 < uMinGrade) ? 1.0 : 0.0;
}`;

const FRAG = `
precision mediump float;
varying vec3 vCol;
varying float vDrop;
void main(){
  if(vDrop > 0.5) discard;
  vec2 d = gl_PointCoord - vec2(0.5);
  if(dot(d, d) > 0.25) discard;          // round points, not squares
  gl_FragColor = vec4(vCol, 1.0);
}`;

const LVERT = `
attribute vec3 aPos;
uniform mat4 uMVP;
void main(){ gl_Position = uMVP * vec4(aPos, 1.0); }`;

const LFRAG = `
precision mediump float;
uniform vec3 uColour;
void main(){ gl_FragColor = vec4(uColour, 1.0); }`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('shader: ' + gl.getShaderInfoLog(s));
  }
  return s;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error('link: ' + gl.getProgramInfoLog(p));
  }
  return p;
}

/* ---- minimal mat4 ---- */
const M4 = {
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  },
  lookAt(eye, tgt, up) {
    const z = norm(sub(eye, tgt));
    const x = norm(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]);
  },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let i = 0; i < 4; i++)
      for (let j = 0; j < 4; j++) {
        let s = 0;
        for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k];
        o[i * 4 + j] = s;
      }
    return o;
  },
};
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => { const m = Math.hypot(...a) || 1; return [a[0] / m, a[1] / m, a[2] / m]; };

export function createViewer(canvas) {
  // preserveDrawingBuffer keeps the frame readable after the draw call, so
  // screenshots and canvas.toDataURL() capture the render instead of a blank
  // buffer. Small cost, and this tool exists to have its results captured.
  const gl = canvas.getContext('webgl', {
    antialias: true, alpha: false, preserveDrawingBuffer: true, depth: true,
  });
  if (!gl) throw new Error('WebGL unavailable');

  const prog = program(gl, VERT, FRAG);
  const lprog = program(gl, LVERT, LFRAG);

  const loc = {
    pos: gl.getAttribLocation(prog, 'aPos'),
    col: gl.getAttribLocation(prog, 'aCol'),
    grd: gl.getAttribLocation(prog, 'aGrade'),
    mvp: gl.getUniformLocation(prog, 'uMVP'),
    ps: gl.getUniformLocation(prog, 'uPointScale'),
    mode: gl.getUniformLocation(prog, 'uMode'),
    ming: gl.getUniformLocation(prog, 'uMinGrade'),
  };
  const lloc = {
    pos: gl.getAttribLocation(lprog, 'aPos'),
    mvp: gl.getUniformLocation(lprog, 'uMVP'),
    col: gl.getUniformLocation(lprog, 'uColour'),
  };

  const buf = { pos: gl.createBuffer(), col: gl.createBuffer(), grd: gl.createBuffer() };
  const camBuf = gl.createBuffer();

  const S = {
    count: 0, camCount: 0,
    yaw: -0.6, pitch: 0.45, dist: 105,
    mode: 0, minGrade: 0, pointScale: 260,
    spin: true, showCams: true,
    drag: false, lx: 0, ly: 0,
    fps: 0, _t: performance.now(), _n: 0,
  };

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, r.width * dpr);
    canvas.height = Math.max(1, r.height * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  canvas.addEventListener('pointerdown', e => {
    S.drag = true; S.spin = false; S.lx = e.clientX; S.ly = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointerup', () => { S.drag = false; });
  canvas.addEventListener('pointermove', e => {
    if (!S.drag) return;
    S.yaw += (e.clientX - S.lx) * 0.006;
    S.pitch = Math.max(-1.45, Math.min(1.45, S.pitch + (e.clientY - S.ly) * 0.005));
    S.lx = e.clientX; S.ly = e.clientY;
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    S.dist = Math.max(12, Math.min(400, S.dist * (1 + Math.sign(e.deltaY) * 0.1)));
  }, { passive: false });

  function upload(data) {
    const pos = new Float32Array(data.positions);
    const n = data.count;
    let col = data.colours && data.colours.length === n * 3
      ? new Float32Array(data.colours)
      : new Float32Array(n * 3).fill(0.8);
    const grd = new Float32Array(n);
    if (data.grades && data.grades.length === n) grd.set(data.grades);
    else grd.fill(3);

    gl.bindBuffer(gl.ARRAY_BUFFER, buf.pos); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.col); gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.grd); gl.bufferData(gl.ARRAY_BUFFER, grd, gl.STATIC_DRAW);
    S.count = n;

    S.camCount = 0;
    if (data.cameras && data.cameras.length >= 6) {
      const c = new Float32Array(data.cameras);
      gl.bindBuffer(gl.ARRAY_BUFFER, camBuf);
      gl.bufferData(gl.ARRAY_BUFFER, c, gl.STATIC_DRAW);
      S.camCount = c.length / 3;
    }
  }

  function frame() {
    if (S.spin) S.yaw += 0.0022;

    gl.clearColor(0.047, 0.047, 0.055, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    const cp = Math.cos(S.pitch), sp = Math.sin(S.pitch);
    const eye = [S.dist * cp * Math.sin(S.yaw), S.dist * cp * Math.cos(S.yaw), S.dist * sp];
    const view = M4.lookAt(eye, [0, 0, 0], [0, 0, 1]);
    const proj = M4.perspective(1.0, canvas.width / canvas.height, 0.5, 4000);
    const mvp = M4.mul(proj, view);

    if (S.count) {
      gl.useProgram(prog);
      gl.uniformMatrix4fv(loc.mvp, false, mvp);
      gl.uniform1f(loc.ps, S.pointScale);
      gl.uniform1f(loc.mode, S.mode);
      gl.uniform1f(loc.ming, S.minGrade);

      gl.bindBuffer(gl.ARRAY_BUFFER, buf.pos);
      gl.enableVertexAttribArray(loc.pos); gl.vertexAttribPointer(loc.pos, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf.col);
      gl.enableVertexAttribArray(loc.col); gl.vertexAttribPointer(loc.col, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf.grd);
      gl.enableVertexAttribArray(loc.grd); gl.vertexAttribPointer(loc.grd, 1, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.POINTS, 0, S.count);
    }

    if (S.showCams && S.camCount > 1) {
      gl.useProgram(lprog);
      gl.uniformMatrix4fv(lloc.mvp, false, mvp);
      gl.uniform3f(lloc.col, 0.788, 0.635, 0.153);
      gl.bindBuffer(gl.ARRAY_BUFFER, camBuf);
      gl.enableVertexAttribArray(lloc.pos);
      gl.vertexAttribPointer(lloc.pos, 3, gl.FLOAT, false, 0, 0);
      gl.drawArrays(gl.LINE_STRIP, 0, S.camCount);
      gl.drawArrays(gl.POINTS, 0, S.camCount);
    }

    S._n++;
    const now = performance.now();
    if (now - S._t > 500) { S.fps = Math.round(S._n * 1000 / (now - S._t)); S._t = now; S._n = 0; }

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  return {
    upload,
    state: S,
    setMode: m => { S.mode = m === 'quality' ? 1 : 0; },
    setMinGrade: g => { S.minGrade = g; },
    setPointScale: v => { S.pointScale = v; },
    toggleSpin: () => (S.spin = !S.spin),
    toggleCams: () => (S.showCams = !S.showCams),
    reset: () => { S.yaw = -0.6; S.pitch = 0.45; S.dist = 105; S.spin = true; },
    fps: () => S.fps,
  };
}
