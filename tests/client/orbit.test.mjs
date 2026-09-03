/* Exercises the real `Ring.frameRing`, `_applyOrbit`, `_bindOrbit` and `resetView` against the
 * vendored three.js, with no browser. The Ring constructor needs WebGL, so the instance is built
 * with Object.create and the two fields those methods touch — the methods themselves are the
 * shipped ones, called through Ring.prototype. */

import * as THREE from './vendor/three.module.min.js';
import { Ring } from './ring.js';

let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) failures++;
};
const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

/* A canvas that records listeners so they can be fired directly. */
function makeRing() {
  const listeners = {};
  const canvas = {
    clientWidth: 960,
    clientHeight: 540,
    addEventListener: (type, fn) => { (listeners[type] ||= []).push(fn); },
    setPointerCapture: () => {},
    releasePointerCapture: () => {},
    hasPointerCapture: () => false,
  };
  const ring = Object.create(Ring.prototype);
  ring.canvas = canvas;
  ring.renderer = { setSize: () => {} };
  ring.camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.05, 200);
  ring.camera.up.set(0, 0, 1);
  ring.orbit = {
    azimuth: -Math.PI / 2,
    elevation: 0,
    distance: 1,
    target: new THREE.Vector3(0, 0, 0.9),
    home: null,
  };
  Ring.prototype._bindOrbit.call(ring);
  const fire = (type, event) => (listeners[type] || []).forEach((fn) => fn(event));
  return { ring, fire };
}

const distance = (camera, target) =>
  Math.hypot(camera.position.x - target.x, camera.position.y - target.y, camera.position.z - target.z);

/* 1. The default framing must be exactly what the fixed camera did before orbiting existed. */
{
  const { ring } = makeRing();
  Ring.prototype.frameRing.call(ring, { ring_size: 4.9 });
  const half = 4.9 / 2;
  const p = ring.camera.position;
  check(
    'home framing reproduces the old fixed camera',
    near(p.x, 0) && near(p.y, -(half + 3.4)) && near(p.z, 2.6),
    `got (${p.x.toFixed(3)}, ${p.y.toFixed(3)}, ${p.z.toFixed(3)}) want (0, ${(-(half + 3.4)).toFixed(3)}, 2.600)`,
  );
}

/* 2. Dragging orbits about the target: the camera moves, its distance to the target does not. */
{
  const { ring, fire } = makeRing();
  Ring.prototype.frameRing.call(ring, { ring_size: 4.9 });
  const before = ring.camera.position.clone();
  const d0 = distance(ring.camera, ring.orbit.target);

  fire('pointerdown', { button: 0, clientX: 400, clientY: 300, pointerId: 1 });
  fire('pointermove', { clientX: 700, clientY: 300, pointerId: 1 });
  fire('pointerup', { pointerId: 1 });

  const moved = ring.camera.position.distanceTo(before);
  const d1 = distance(ring.camera, ring.orbit.target);
  check('drag moves the camera', moved > 0.5, `moved ${moved.toFixed(3)} m`);
  check('drag keeps the orbit radius', near(d0, d1, 1e-9), `${d0.toFixed(6)} -> ${d1.toFixed(6)}`);
  check('drag right turns the ring right', ring.orbit.azimuth < -Math.PI / 2);
}

/* 3. The wheel zooms multiplicatively and stays inside its clamps. */
{
  const { ring, fire } = makeRing();
  Ring.prototype.frameRing.call(ring, { ring_size: 4.9 });
  const d0 = distance(ring.camera, ring.orbit.target);

  fire('wheel', { deltaY: -400, preventDefault: () => {} });
  const closer = distance(ring.camera, ring.orbit.target);
  check('wheel up moves closer', closer < d0 - 0.1, `${d0.toFixed(2)} -> ${closer.toFixed(2)} m`);

  for (let i = 0; i < 200; i++) fire('wheel', { deltaY: -1000, preventDefault: () => {} });
  check('zoom clamps near', ring.orbit.distance >= 1.5 - 1e-9, `${ring.orbit.distance.toFixed(3)} m`);
  for (let i = 0; i < 400; i++) fire('wheel', { deltaY: 1000, preventDefault: () => {} });
  check('zoom clamps far', ring.orbit.distance <= 24.0 + 1e-9, `${ring.orbit.distance.toFixed(3)} m`);
}

/* 4. Elevation clamps short of the pole rather than flipping the view over. */
{
  const { ring, fire } = makeRing();
  Ring.prototype.frameRing.call(ring, { ring_size: 4.9 });
  fire('pointerdown', { button: 0, clientX: 400, clientY: 300, pointerId: 1 });
  fire('pointermove', { clientX: 400, clientY: -5000, pointerId: 1 });
  fire('pointerup', { pointerId: 1 });
  check('elevation stays below vertical', ring.orbit.elevation < Math.PI / 2, `${ring.orbit.elevation.toFixed(3)} rad`);
  check('camera stays above what it looks at', ring.camera.position.z > ring.orbit.target.z);

  fire('pointerdown', { button: 0, clientX: 400, clientY: 300, pointerId: 1 });
  fire('pointermove', { clientX: 400, clientY: 5000, pointerId: 1 });
  fire('pointerup', { pointerId: 1 });
  check('elevation stays above the floor', ring.orbit.elevation > 0, `${ring.orbit.elevation.toFixed(3)} rad`);
}

/* 5. Double-click returns exactly to the framing frameRing chose. */
{
  const { ring, fire } = makeRing();
  Ring.prototype.frameRing.call(ring, { ring_size: 4.9 });
  const home = ring.camera.position.clone();

  fire('pointerdown', { button: 0, clientX: 400, clientY: 300, pointerId: 1 });
  fire('pointermove', { clientX: 900, clientY: 100, pointerId: 1 });
  fire('pointerup', { pointerId: 1 });
  fire('wheel', { deltaY: -600, preventDefault: () => {} });
  check('view actually left home', ring.camera.position.distanceTo(home) > 0.5);

  fire('dblclick', {});
  check(
    'double-click restores home exactly',
    ring.camera.position.distanceTo(home) < 1e-9,
    `off by ${ring.camera.position.distanceTo(home).toExponential(2)} m`,
  );
}

/* 6. A non-left button must not start a drag. */
{
  const { ring, fire } = makeRing();
  Ring.prototype.frameRing.call(ring, { ring_size: 4.9 });
  const before = ring.camera.position.clone();
  fire('pointerdown', { button: 2, clientX: 400, clientY: 300, pointerId: 1 });
  fire('pointermove', { clientX: 900, clientY: 300, pointerId: 1 });
  check('right-button drag is ignored', ring.camera.position.distanceTo(before) < 1e-9);
}

/* 7. A ring of a different size still frames correctly. */
{
  const { ring } = makeRing();
  Ring.prototype.frameRing.call(ring, { ring_size: 7.0 });
  const p = ring.camera.position;
  check(
    'framing follows ring size',
    near(p.x, 0) && near(p.y, -(3.5 + 3.4)) && near(p.z, 2.6),
    `got y=${p.y.toFixed(3)} want ${(-(3.5 + 3.4)).toFixed(3)}`,
  );
}

console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILED`);
process.exit(failures === 0 ? 0 : 1);
