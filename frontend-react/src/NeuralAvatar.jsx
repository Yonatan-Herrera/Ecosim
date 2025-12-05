import React, { useEffect, useRef } from 'react';

const NeuralAvatar = ({
    active = true,
    mood = 'neutral',
    variant = 'human',
    width = 300,
    height = 400
}) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        // Handle high-DPI displays for crisp rendering
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;

        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);

        let animationFrameId;

        const buildHumanGeometry = () => {
            const points = [];
            const addPoint = (x, y, z, tag) => points.push({ x, y, z, tag });
            const S = 6.5;

            for (let i = 0; i < 25; i++) {
                const theta = Math.random() * Math.PI * 2;
                const phi = Math.random() * Math.PI;
                const r = 14 * S;
                addPoint(
                    r * Math.sin(phi) * Math.cos(theta),
                    r * Math.sin(phi) * Math.sin(theta) - (65 * S),
                    r * Math.cos(phi),
                    'head'
                );
            }

            for (let i = 0; i < 45; i++) {
                const theta = Math.random() * Math.PI * 2;
                const y = ((Math.random() * 65) - 45) * S;
                const radiusAtHeight = (12 * S) + (y / 10);
                const r = Math.random() * radiusAtHeight;
                addPoint(r * Math.cos(theta), y, r * Math.sin(theta), 'body');
            }

            [-22 * S, 22 * S].forEach(xOffset => {
                for (let y = -45 * S; y < 15 * S; y += 6 * S) {
                    addPoint(xOffset + (Math.random() * 4 - 2) * S, y, (Math.random() * 6 - 3) * S, 'arm');
                }
            });

            [-10 * S, 10 * S].forEach(xOffset => {
                for (let y = 20 * S; y < 90 * S; y += 7 * S) {
                    addPoint(xOffset + (Math.random() * 4 - 2) * S, y, (Math.random() * 6 - 3) * S, 'leg');
                }
            });

            return {
                points,
                connectionRadius: 16 * S,
                rotationSpeed: 0.008,
                palette: {
                    link: 'rgba(20, 184, 166, 0.15)',
                    happy: 'rgba(45, 212, 191, 0.25)',
                    nodePrimary: '#0d9488',
                    nodeAccent: '#2dd4bf'
                }
            };
        };

        const buildBuildingGeometry = () => {
            const points = [];
            const addPoint = (x, y, z, tag) => points.push({ x, y, z, tag });
            const heightSpan = 320;
            const floors = 45;
            const baseWidth = 80;
            const baseDepth = 50;

            for (let f = 0; f <= floors; f++) {
                const t = f / floors;
                const width = baseWidth * (1 - t * 0.4);
                const depth = baseDepth * (1 - t * 0.3);
                const y = (t - 0.5) * heightSpan;

                const corners = [
                    [width, y, depth],
                    [-width, y, depth],
                    [-width, y, -depth],
                    [width, y, -depth]
                ];

                corners.forEach(([x, yPos, z]) => addPoint(x, yPos, z, 'frame'));

                for (let w = 0; w < 6; w++) {
                    const offsetY = y + (Math.random() - 0.5) * (heightSpan / floors);
                    const offsetX = (Math.random() * 2 - 1) * width * 0.85;
                    addPoint(offsetX, offsetY, depth + 4, 'window');
                    addPoint(offsetX, offsetY, -depth - 4, 'window');
                }

                for (let w = 0; w < 4; w++) {
                    const offsetY = y + (Math.random() - 0.5) * (heightSpan / floors);
                    const offsetZ = (Math.random() * 2 - 1) * depth * 0.85;
                    addPoint(width + 4, offsetY, offsetZ, 'window');
                    addPoint(-width - 4, offsetY, offsetZ, 'window');
                }
            }

            return {
                points,
                connectionRadius: 40,
                rotationSpeed: 0.004,
                palette: {
                    link: 'rgba(14, 165, 233, 0.22)',
                    happy: 'rgba(56, 189, 248, 0.35)',
                    nodePrimary: '#0ea5e9',
                    nodeAccent: '#38bdf8'
                }
            };
        };

        const geometry = variant === 'building' ? buildBuildingGeometry() : buildHumanGeometry();
        const { points, connectionRadius, rotationSpeed, palette } = geometry;

        const connections = [];
        points.forEach((p1, i) => {
            points.forEach((p2, j) => {
                if (i === j) return;
                const dist = Math.sqrt(
                    Math.pow(p1.x - p2.x, 2) +
                    Math.pow(p1.y - p2.y, 2) +
                    Math.pow(p1.z - p2.z, 2)
                );
                if (dist < connectionRadius) {
                    connections.push([i, j]);
                }
            });
        });

        let angle = 0;

        const render = () => {
            if (!active) return;

            // Clear with slight trail effect (optional, currently full clear)
            ctx.clearRect(0, 0, rect.width, rect.height);

            const centerX = rect.width / 2;
            const centerY = rect.height / 2;

            angle += rotationSpeed;
            const cos = Math.cos(angle);
            const sin = Math.sin(angle);

            // --- 3. PROJECTION LOOP ---
            const projected = points.map(p => {
                // Rotate around Y-axis
                const x = p.x * cos - p.z * sin;
                const z = p.x * sin + p.z * cos;
                const y = p.y; // Y stays same

                // Simple Perspective Projection
                const fov = 350;
                const scale = fov / (fov + z + 200); // Camera distance

                return {
                    x: x * scale + centerX,
                    y: y * scale + centerY,
                    scale: scale,
                    tag: p.tag
                };
            });

            // --- 4. DRAWING ---

            // Draw Connections (Synapses) first
            ctx.lineWidth = 1;
            ctx.strokeStyle = mood === 'happy' ? palette.happy : palette.link;

            connections.forEach(([i, j]) => {
                const p1 = projected[i];
                const p2 = projected[j];

                // Optimization: Only draw if points are large enough (close to camera)
                if (p1.scale > 0.4 && p2.scale > 0.4) {
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
            });

            // Draw Nodes (Neurons)
            projected.forEach(p => {
                const size = Math.max(0.5, 2.5 * p.scale);
                ctx.beginPath();
                ctx.arc(p.x, p.y, size, 0, Math.PI * 2);

                // Flicker effect for neural activity
                const alpha = 0.35 + Math.random() * 0.5;

                ctx.fillStyle = (p.tag === 'head' || p.tag === 'window')
                    ? palette.nodeAccent
                    : palette.nodePrimary;
                ctx.globalAlpha = alpha;
                ctx.shadowBlur = p.tag === 'window' ? 10 : 0;
                ctx.shadowColor = palette.nodeAccent;

                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.shadowBlur = 0;
            });

            animationFrameId = requestAnimationFrame(render);
        };

        render();

        return () => cancelAnimationFrame(animationFrameId);
    }, [active, mood, variant, width, height]);

    return (
        <canvas
            ref={canvasRef}
            style={{ width: '100%', height: '100%' }}
            className="block"
        />
    );
};

export default NeuralAvatar;
