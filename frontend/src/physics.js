// ─────────────────────────────────────────────────
//  physics.js  —  Spring + SliderPhysics + SwitchBehavior
//  Ported from TypeGPU Spring.ts / Switch.ts / Slider.ts
// ─────────────────────────────────────────────────

export class Spring {
  constructor({ mass = 1, stiffness = 1000, damping = 10 }) {
    this.mass = mass; this.stiffness = stiffness; this.damping = damping;
    this.value = 0; this.target = 0; this.velocity = 0;
  }
  update(dt) {
    const a = (-this.stiffness * (this.value - this.target) - this.damping * this.velocity) / this.mass;
    this.velocity += a * dt;
    this.value    += this.velocity * dt;
  }
}

export class SwitchBehavior {
  constructor() {
    this.toggled = false; this.pressed = false;
    this._p = 0; this._v = 0; this.A = 100;
    this._sX = new Spring({ mass:1, stiffness:1000, damping:10 });
    this._sZ = new Spring({ mass:1, stiffness:900,  damping:12 });
    this._wX = new Spring({ mass:1, stiffness:1000, damping:20 });
  }
  get state() {
    return { progress: this._p, squashX: this._sX.value, squashZ: this._sZ.value, wiggleX: this._wX.value };
  }
  update(dt) {
    if (dt <= 0) return;
    let acc = 0;
    if ( this.toggled && this._p < 1) acc =  this.A;
    if (!this.toggled && this._p > 0) acc = -this.A;
    if (this.pressed) {
      this._sX.velocity = -2;
      this._sZ.velocity =  1;
      this._wX.velocity =  1 * Math.sign(this._p - 0.5);
    }
    this._v += acc * dt;
    if (this._p > 0 && this._p < 1) this._wX.velocity = this._v;
    this._p += this._v * dt;
    if (this._p > 1) { this._p=1; this._v=0; this._sX.velocity=-5; this._sZ.velocity=5; this._wX.velocity=-10; }
    if (this._p < 0) { this._p=0; this._v=0; this._sX.velocity=-5; this._sZ.velocity=5; this._wX.velocity= 10; }
    this._p = Math.max(0, Math.min(1, this._p));
    this._sX.update(dt); this._sZ.update(dt); this._wX.update(dt);
  }
}

function smoothstep(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

export class SliderPhysics {
  constructor(n = 13, len) {
    this.n = n; this.len = len; this.rest = len / (n - 1);
    this.damping = 0.015; this.iters = 16; this.subs = 6;
    this.bendK = 0.1; this.archK = 2; this.flatN = 1; this.flatK = 0.05;
    this.bendE = 1.2; this.edgeD = 0.01; this.anchor = 0;
    this._p = []; this._q = [];
    this._m = new Float32Array(n).fill(1); this._m[0] = 0; this._m[n-1] = 0;
    this._tx = len;
    this._reset();
  }
  _reset() {
    this._p = Array.from({ length: this.n }, (_, i) => ({ x: this.anchor + (i / (this.n-1)) * this.len, y: 0 }));
    this._q = this._p.map(p => ({ ...p }));
  }
  init(anchor) { this.anchor = anchor; this._tx = anchor + this.len; this._reset(); }
  setX(x) { this._tx = Math.max(this.anchor, Math.min(this.anchor + this.len, x)); }
  update(dt) {
    if (dt <= 0) return;
    const h = dt / this.subs, d = Math.min(this.damping, 0.999);
    const comp = Math.max(0, 1 - Math.abs(this._tx - this.anchor) / this.len);
    for (let s = 0; s < this.subs; s++) { this._integrate(h, d, comp); this._constrain(); }
  }
  _integrate(h, d, comp) {
    for (let i = 0; i < this.n; i++) {
      if (i === 0)        { this._p[i] = { x: this.anchor, y: 0 }; this._q[i] = { x: this.anchor, y: 0 }; continue; }
      if (i === this.n-1) { this._p[i] = { x: this._tx, y: 0 };    this._q[i] = { x: this._tx, y: 0 };    continue; }
      const { x: px, y: py } = this._p[i];
      const vx = (px - this._q[i].x) * (1 - d), vy = (py - this._q[i].y) * (1 - d);
      let ay = 0;
      if (comp > 0) {
        const t = i / (this.n-1), e = this.edgeD;
        ay = this.archK * Math.sin(Math.PI * t) * smoothstep(e, 1-e, t) * smoothstep(e, 1-e, 1-t) * comp;
      }
      this._q[i] = { x: px, y: py };
      this._p[i] = { x: px + vx, y: Math.max(0, py + vy + ay * h * h) };
    }
  }
  _constrain() {
    for (let it = 0; it < this.iters; it++) {
      for (let i = 0; i < this.n-1; i++) this._pd(i, i+1, this.rest, 0.1);
      for (let i = 1; i < this.n-1; i++) {
        const t = i / (this.n-1);
        const k = this.bendK * (0.05 + 0.95 * Math.pow(Math.abs(t - 0.5) * 2, this.bendE));
        this._pd(i-1, i+1, 2 * this.rest, k);
      }
      if (this.flatN > 0) {
        const c = Math.min(this.flatN, this.n-2);
        for (let i = 1; i <= c; i++) this._py(i);
        for (let i = this.n-1-c; i < this.n-1; i++) this._py(i);
      }
      this._p[0]      = { x: this.anchor, y: 0 };
      this._p[this.n-1] = { x: this._tx, y: 0 };
    }
  }
  _pd(i, j, rest, k = 1) {
    const dx = this._p[j].x - this._p[i].x, dy = this._p[j].y - this._p[i].y;
    const len = Math.hypot(dx, dy); if (len < 1e-8) return;
    const w1 = this._m[i], w2 = this._m[j], ws = w1 + w2; if (ws <= 0) return;
    const df = (len - rest) / len;
    this._p[i].x += dx * df * (w1/ws) * k; this._p[i].y += dy * df * (w1/ws) * k;
    this._p[j].x -= dx * df * (w2/ws) * k; this._p[j].y -= dy * df * (w2/ws) * k;
  }
  _py(i) { if (i <= 0 || i >= this.n-1 || this._m[i] <= 0) return; this._p[i].y += (0 - this._p[i].y) * Math.min(this.flatK, 1); }
  get pts() { return this._p; }
}
