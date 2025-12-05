import React, { useEffect, useRef } from 'react';

const NeuralBuilding = ({
  active = true,
  activityLevel = 'normal',
  width = 300,
  height = 400,
  tier = 3
}) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    let animationFrameId;

    const points = [];
    const connections = [];

    const addPoint = (x, y, z, tag) => points.push({ x, y, z, tag });

    const createBlock = (yBottom, yTop, widthSize, depthSize) => {
      [-widthSize / 2, widthSize / 2].forEach(x => {
        [-depthSize / 2, depthSize / 2].forEach(z => {
          for (let y = yBottom; y <= yTop; y += 10) {
            addPoint(x, y, z, 'structure');
          }
        });
      });

      for (let y = yBottom; y <= yTop; y += 15) {
        for (let x = -widthSize / 2; x <= widthSize / 2; x += widthSize / 2) {
          for (let z = -depthSize / 2; z <= depthSize / 2; z += 5) {
            if (Math.abs(x) === widthSize / 2) addPoint(x, y, z, 'window');
          }
        }
        for (let z = -depthSize / 2; z <= depthSize / 2; z += depthSize / 2) {
          for (let x = -widthSize / 2; x <= widthSize / 2; x += 5) {
            if (Math.abs(z) === depthSize / 2) addPoint(x, y, z, 'window');
          }
        }
      }
    };

    const baseW = tier === 3 ? 50 : tier === 2 ? 40 : 30;
    const startY = 80;

    createBlock(startY - 40, startY, baseW, baseW);
    if (tier >= 2) {
      createBlock(startY - 90, startY - 40, baseW * 0.8, baseW * 0.8);
    } else {
      createBlock(startY - 60, startY - 40, baseW, baseW);
    }
    if (tier >= 3) {
      createBlock(startY - 150, startY - 90, baseW * 0.6, baseW * 0.6);
    }

    for (let y = startY - tier * 55; y < startY; y += 4) {
      addPoint(0, y, 0, 'core');
    }

    if (tier >= 2) {
      const topY = startY - (tier === 3 ? 150 : 90);
      for (let y = topY - 30; y < topY; y += 5) {
        addPoint(0, y, 0, 'spire');
      }
    }

    points.forEach((p1, i) => {
      points.forEach((p2, j) => {
        if (i === j) return;
        const dx = Math.abs(p1.x - p2.x);
        const dy = Math.abs(p1.y - p2.y);
        const dz = Math.abs(p1.z - p2.z);

        let connected = false;
        if (dx < 1 && dz < 1 && dy < 16) connected = true;
        if (dy < 1) {
          if (dx < 1 && dz < 10) connected = true;
          if (dz < 1 && dx < 10) connected = true;
        }
        if (p1.tag === 'core' && p2.tag === 'core' && dy < 6) connected = true;

        if (connected) {
          connections.push([i, j]);
        }
      });
    });

    const bounds = points.reduce(
      (acc, p) => ({
        minX: Math.min(acc.minX, p.x),
        maxX: Math.max(acc.maxX, p.x),
        minY: Math.min(acc.minY, p.y),
        maxY: Math.max(acc.maxY, p.y),
        minZ: Math.min(acc.minZ, p.z),
        maxZ: Math.max(acc.maxZ, p.z)
      }),
      { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity }
    );

    const modelHeight = bounds.maxY - bounds.minY || 1;
    const modelWidth = bounds.maxX - bounds.minX || 1;
    const modelDepth = bounds.maxZ - bounds.minZ || 1;
    const targetHeight = rect.height * 0.9;
    const targetWidth = rect.width * 0.7;
    const scaleFactor = Math.min(targetHeight / modelHeight, targetWidth / Math.max(modelWidth, modelDepth));

    const centerXModel = (bounds.minX + bounds.maxX) / 2;
    const centerYModel = (bounds.minY + bounds.maxY) / 2;
    const centerZModel = (bounds.minZ + bounds.maxZ) / 2;

    points.forEach(p => {
      p.sx = (p.x - centerXModel) * scaleFactor;
      p.sy = (p.y - centerYModel) * scaleFactor;
      p.sz = (p.z - centerZModel) * scaleFactor;
    });

    let angle = 0;
    let pulse = 0;

    const render = () => {
      if (!active) return;

      ctx.clearRect(0, 0, rect.width, rect.height);
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;

      angle += 0.003;
      pulse += 0.05;

      const cos = Math.cos(angle);
      const sin = Math.sin(angle);

      const projected = points.map(p => {
        const x = p.sx * cos - p.sz * sin;
        const z = p.sx * sin + p.sz * cos;
        const y = p.sy;

        const fov = 400;
        const scale = fov / (fov + z + 200);

        return {
          x: x * scale + centerX,
          y: y * scale + centerY,
          scale,
          tag: p.tag,
          origY: p.y
        };
      });

      ctx.lineWidth = 1;
      const baseAlpha = activityLevel === 'high' ? 0.4 : 0.2;
      const pulseColor = activityLevel === 'high' ? '200, 250, 255' : '45, 212, 191';
      ctx.strokeStyle = `rgba(20, 184, 166, ${baseAlpha})`;

      connections.forEach(([i, j]) => {
        const p1 = projected[i];
        const p2 = projected[j];

        if (p1.scale > 0.25 && p2.scale > 0.25) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);

          if (p1.tag === 'core') {
            ctx.save();
            ctx.strokeStyle = `rgba(${pulseColor}, 0.8)`;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();
          } else {
            ctx.stroke();
          }
        }
      });

      projected.forEach(p => {
        let size = Math.max(0.5, 2 * p.scale);
        ctx.beginPath();

        let r = 20, g = 184, b = 166, a = 0.3;

        if (p.tag === 'core') {
          const wave = Math.sin(p.origY / 10 - pulse);
          if (wave > 0.7) {
            r = 255; g = 255; b = 255; a = 1;
            size *= 1.2;
            ctx.shadowBlur = 8;
            ctx.shadowColor = 'white';
          } else {
            r = 45; g = 212; b = 191; a = 0.5;
            ctx.shadowBlur = 0;
          }
        } else if (p.tag === 'window') {
          if (Math.random() > 0.99) {
            r = 200; g = 240; b = 255; a = 0.9;
            ctx.shadowBlur = 4;
            ctx.shadowColor = '#bae6fd';
          } else {
            a = 0.1;
            ctx.shadowBlur = 0;
          }
          size *= 0.8;
        } else if (p.tag === 'spire') {
          r = 239; g = 68; b = 68; a = Math.floor(pulse / 5) % 2 === 0 ? 0.2 : 0.8;
          ctx.shadowBlur = 10;
          ctx.shadowColor = 'red';
        } else {
          ctx.shadowBlur = 0;
        }

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${a})`;
        const dim = size * 2;
        ctx.fillRect(p.x - size, p.y - size, dim, dim);
        ctx.shadowBlur = 0;
      });

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animationFrameId);
  }, [active, activityLevel, width, height, tier]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%' }}
      className="block"
    />
  );
};

export default NeuralBuilding;
