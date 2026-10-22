"""
3D WebGL (Three.js) Interactive Federated Learning Simulation Visualizer.
Renders 4 Decentralized Hospital Nodes at the top and 1 Central Global Model Server below,
with live flowing particle beams, rotating orbital gyroscope rings, and interactive 3D controls.
"""

def get_3d_simulation_html(phase_name="Active Telemetry", round_num=1, total_rounds=5, active_hospitals=None):
    if active_hospitals is None:
        active_hospitals = [1, 2, 3, 4]
        
    act_h1 = 1 in active_hospitals
    act_h2 = 2 in active_hospitals
    act_h3 = 3 in active_hospitals
    act_h4 = 4 in active_hospitals

    html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            background: linear-gradient(135deg, #070D1E 0%, #0F172A 100%);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            user-select: none;
        }}
        #canvas3d {{
            width: 100vw;
            height: 480px;
            display: block;
        }}
        .hud-panel {{
            position: absolute;
            top: 14px;
            left: 14px;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 10px;
            padding: 10px 16px;
            color: #F8FAFC;
            font-size: 13px;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            pointer-events: none;
        }}
        .hud-title {{
            font-size: 13px;
            font-weight: 700;
            color: #38BDF8;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .hud-badge {{
            display: inline-block;
            background: #0284C7;
            color: #FFFFFF;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .controls-panel {{
            position: absolute;
            top: 14px;
            right: 14px;
            display: flex;
            gap: 8px;
        }}
        .ctrl-btn {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #E2E8F0;
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .ctrl-btn:hover {{
            background: #0284C7;
            color: #FFFFFF;
            border-color: #38BDF8;
        }}
        .legend-panel {{
            position: absolute;
            bottom: 12px;
            left: 14px;
            display: flex;
            gap: 16px;
            font-size: 11px;
            color: #94A3B8;
            background: rgba(15, 23, 42, 0.7);
            padding: 6px 12px;
            border-radius: 6px;
            pointer-events: none;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .dot-green {{ width: 8px; height: 8px; border-radius: 50%; background: #10B981; }}
        .dot-red {{ width: 8px; height: 8px; border-radius: 50%; background: #EF4444; }}
        .dot-cyan {{ width: 8px; height: 8px; border-radius: 50%; background: #38BDF8; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div class="hud-panel">
        <div class="hud-title">🌐 3D FEDERATED MEDICAL NETWORK SIMULATION</div>
        <div>Round: <b>{round_num}/{total_rounds}</b> | Phase: <span class="hud-badge" id="phaseBadge">{phase_name}</span></div>
    </div>

    <div class="controls-panel">
        <button class="ctrl-btn" onclick="togglePlay()">⏯ Play / Pause</button>
        <button class="ctrl-btn" onclick="nextPhase()">🔄 Next Phase</button>
        <button class="ctrl-btn" onclick="resetCam()">🎯 Reset Camera</button>
    </div>

    <div class="legend-panel">
        <div class="legend-item"><div class="dot-green"></div> Active Hospital Node</div>
        <div class="legend-item"><div class="dot-red"></div> Network Timeout Node</div>
        <div class="legend-item"><div class="dot-cyan"></div> Central Global Model Server (Below)</div>
        <div class="legend-item"><span>🖱️ Click & Drag to Rotate | Scroll to Zoom</span></div>
    </div>

    <div id="canvas3d"></div>

    <script>
        const container = document.getElementById('canvas3d');
        const width = window.innerWidth;
        const height = 480;

        // 1. Scene & Camera Setup
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x070D1E, 0.025);

        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 0.5, 17);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxDistance = 30;
        controls.minDistance = 6;

        // 2. Lighting
        const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.7);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0x38BDF8, 1.2);
        dirLight.position.set(0, 15, 10);
        scene.add(dirLight);

        // Core light at Global Model
        const coreLight = new THREE.PointLight(0x00F2FE, 2.5, 20);
        coreLight.position.set(0, -3.2, 0);
        scene.add(coreLight);

        // 3. Background Cyber Grid
        const gridHelper = new THREE.GridHelper(30, 30, 0x1E3A8A, 0x0F172A);
        gridHelper.position.y = -5.0;
        scene.add(gridHelper);

        // Text Sprite Helper
        function createTextSprite(text, color, subtitle) {{
            const canvas = document.createElement('canvas');
            canvas.width = 512;
            canvas.height = 160;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
            ctx.roundRect ? ctx.roundRect(10, 10, 492, 140, 16) : ctx.rect(10, 10, 492, 140);
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.lineWidth = 4;
            ctx.stroke();

            ctx.fillStyle = color;
            ctx.font = 'bold 36px Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(text, 256, 68);

            if (subtitle) {{
                ctx.fillStyle = '#94A3B8';
                ctx.font = '22px Arial, sans-serif';
                ctx.fillText(subtitle, 256, 115);
            }}

            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({{ map: texture, transparent: true }});
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(3.2, 1.0, 1);
            return sprite;
        }}

        // 4. Central Global Model Server (Positioned BELOW at y = -3.2)
        const serverGroup = new THREE.Group();
        serverGroup.position.set(0, -3.2, 0);

        // Core Sphere
        const coreGeo = new THREE.SphereGeometry(1.25, 32, 32);
        const coreMat = new THREE.MeshPhongMaterial({{
            color: 0x0284C7,
            emissive: 0x0369A1,
            shininess: 90,
            wireframe: false,
            transparent: true,
            opacity: 0.95
        }});
        const coreMesh = new THREE.Mesh(coreGeo, coreMat);
        serverGroup.add(coreMesh);

        // Inner glowing wireframe
        const wireGeo = new THREE.SphereGeometry(1.30, 16, 16);
        const wireMat = new THREE.MeshBasicMaterial({{ color: 0x38BDF8, wireframe: true, transparent: true, opacity: 0.4 }});
        const wireMesh = new THREE.Mesh(wireGeo, wireMat);
        serverGroup.add(wireMesh);

        // Concentric Orbital Gyroscope Rings (Representing Tensor Aggregation)
        const ringMat1 = new THREE.MeshBasicMaterial({{ color: 0x00F2FE, side: THREE.DoubleSide }});
        const ringGeo1 = new THREE.TorusGeometry(1.9, 0.04, 16, 100);
        const ringMesh1 = new THREE.Mesh(ringGeo1, ringMat1);
        serverGroup.add(ringMesh1);

        const ringMat2 = new THREE.MeshBasicMaterial({{ color: 0x818CF8, side: THREE.DoubleSide }});
        const ringGeo2 = new THREE.TorusGeometry(2.3, 0.04, 16, 100);
        const ringMesh2 = new THREE.Mesh(ringGeo2, ringMat2);
        ringMesh2.rotation.x = Math.PI / 3;
        serverGroup.add(ringMesh2);

        const ringMat3 = new THREE.MeshBasicMaterial({{ color: 0x38BDF8, side: THREE.DoubleSide }});
        const ringGeo3 = new THREE.TorusGeometry(2.7, 0.04, 16, 100);
        const ringMesh3 = new THREE.Mesh(ringGeo3, ringMat3);
        ringMesh3.rotation.y = Math.PI / 4;
        serverGroup.add(ringMesh3);

        // Server Label Sprite
        const serverLabel = createTextSprite("🌐 CENTRAL GLOBAL MODEL", "#38BDF8", "[FedAvg Aggregation Server]");
        serverLabel.position.set(0, -2.1, 0);
        serverGroup.add(serverLabel);

        scene.add(serverGroup);

        // 5. 4 Decentralized Hospital Nodes (Positioned AT TOP)
        const hospitalsData = [
            {{ name: "Hospital A", subtitle: "Central Medical (482 X-Rays)", pos: [-6.2, 3.8, 0], active: {str(act_h1).lower()} }},
            {{ name: "Hospital B", subtitle: "Metro Clinic (420 X-Rays)", pos: [-2.1, 4.4, 0.8], active: {str(act_h2).lower()} }},
            {{ name: "Hospital C", subtitle: "Diagnostic Lab (390 X-Rays)", pos: [2.1, 4.4, 0.8], active: {str(act_h3).lower()} }},
            {{ name: "Hospital D", subtitle: "Regional Hospital (266 X-Rays)", pos: [6.2, 3.8, 0], active: {str(act_h4).lower()} }}
        ];

        const hospitalNodes = [];
        const connectionCurves = [];

        hospitalsData.forEach((h, idx) => {{
            const hGroup = new THREE.Group();
            hGroup.position.set(...h.pos);

            const nodeColor = h.active ? 0x10B981 : 0xEF4444;
            const beaconColor = h.active ? 0x059669 : 0xDC2626;

            // Building / Node Geometry
            const bGeo = new THREE.BoxGeometry(1.2, 1.2, 1.2);
            const bMat = new THREE.MeshPhongMaterial({{
                color: 0x1E293B,
                emissive: h.active ? 0x064E3B : 0x7F1D1D,
                shininess: 80
            }});
            const bMesh = new THREE.Mesh(bGeo, bMat);
            hGroup.add(bMesh);

            // Glowing Wireframe border
            const bWireGeo = new THREE.BoxGeometry(1.25, 1.25, 1.25);
            const bWireMat = new THREE.MeshBasicMaterial({{ color: nodeColor, wireframe: true }});
            hGroup.add(new THREE.Mesh(bWireGeo, bWireMat));

            // Floating Medical Cross on top (+)
            const crossMat = new THREE.MeshBasicMaterial({{ color: nodeColor }});
            const crossH = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.2, 0.2), crossMat);
            crossH.position.y = 1.0;
            const crossV = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.7, 0.2), crossMat);
            crossV.position.y = 1.0;
            hGroup.add(crossH);
            hGroup.add(crossV);

            // Ground beacon ring
            const beaconRing = new THREE.Mesh(
                new THREE.RingGeometry(0.8, 1.0, 32),
                new THREE.MeshBasicMaterial({{ color: nodeColor, side: THREE.DoubleSide, transparent: true, opacity: 0.6 }})
            );
            beaconRing.rotation.x = Math.PI / 2;
            beaconRing.position.y = -0.7;
            hGroup.add(beaconRing);

            // Hospital Label Sprite
            const statusLabel = h.active ? "[Active • Local SGD]" : "[Connection Lost • Timeout]";
            const hSprite = createTextSprite(h.name, h.active ? "#10B981" : "#EF4444", statusLabel);
            hSprite.position.set(0, 1.9, 0);
            hGroup.add(hSprite);

            scene.add(hGroup);
            hospitalNodes.push({{ group: hGroup, data: h, beacon: beaconRing, crossH, crossV }});

            // 6. 3D Connection Curves from Hospital Node -> Central Global Model (Below)
            const startPt = new THREE.Vector3(...h.pos);
            const endPt = new THREE.Vector3(0, -2.0, 0);
            const midPt = new THREE.Vector3(h.pos[0] * 0.45, 0.5, h.pos[2] * 0.45);
            const curve = new THREE.QuadraticBezierCurve3(startPt, midPt, endPt);
            connectionCurves.push({{ curve, active: h.active, color: nodeColor }});

            // Connection Tube Line
            const points = curve.getPoints(50);
            const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineDashedMaterial({{
                color: h.active ? 0x0284C7 : 0x7F1D1D,
                dashSize: h.active ? 0.3 : 0.15,
                gapSize: h.active ? 0.1 : 0.2,
                linewidth: 2,
                transparent: true,
                opacity: h.active ? 0.7 : 0.3
            }});
            const lineMesh = new THREE.Line(lineGeo, lineMat);
            lineMesh.computeLineDistances();
            scene.add(lineMesh);
        }});

        // 7. Dynamic Flowing Particle System (Encrypted Model Updates)
        const particleCount = 120;
        const particleGeo = new THREE.BufferGeometry();
        const pPos = new Float32Array(particleCount * 3);
        const pProgress = new Float32Array(particleCount);
        const pCurveIdx = new Int32Array(particleCount);
        const pSpeed = new Float32Array(particleCount);

        for (let i = 0; i < particleCount; i++) {{
            pCurveIdx[i] = i % 4;
            pProgress[i] = Math.random();
            pSpeed[i] = 0.006 + Math.random() * 0.008;
            pPos[i*3] = 0;
            pPos[i*3+1] = 0;
            pPos[i*3+2] = 0;
        }}
        particleGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));

        const pMat = new THREE.PointsMaterial({{
            color: 0x38BDF8,
            size: 0.28,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending
        }});
        const particleSystem = new THREE.Points(particleGeo, pMat);
        scene.add(particleSystem);

        // 8. Expanding Broadcast Shockwave Ring
        const shockwaveGeo = new THREE.RingGeometry(0.2, 0.5, 64);
        const shockwaveMat = new THREE.MeshBasicMaterial({{
            color: 0x00F2FE,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.0
        }});
        const shockwave = new THREE.Mesh(shockwaveGeo, shockwaveMat);
        shockwave.rotation.x = Math.PI / 2;
        shockwave.position.set(0, -3.2, 0);
        scene.add(shockwave);

        // 9. Simulation Animation Loop & State Management
        let isPlaying = true;
        let phase = 0; // 0: Local SGD, 1: DNA Encrypted Upload, 2: FedAvg Aggregation, 3: Global Broadcast
        const phaseNames = [
            "1. Local SGD Training (FedProx Regularization)",
            "2. DNA Weight Encryption & Telemetry Upload",
            "3. Central Server FedAvg Parameter Aggregation",
            "4. Global Model Broadcast to Hospital Nodes"
        ];
        let phaseTimer = 0;

        function updatePhaseBadge() {{
            const badge = document.getElementById('phaseBadge');
            if (badge) badge.innerText = phaseNames[phase];
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
        }}

        function nextPhase() {{
            phase = (phase + 1) % 4;
            phaseTimer = 0;
            updatePhaseBadge();
        }}

        function resetCam() {{
            camera.position.set(0, 0.5, 17);
            controls.target.set(0, 0, 0);
        }}

        let shockwaveScale = 1;

        function animate() {{
            requestAnimationFrame(animate);

            // Controls update
            controls.update();

            // Rotate Central Global Model Gyroscope Rings
            ringMesh1.rotation.z += 0.020;
            ringMesh2.rotation.y += 0.015;
            ringMesh3.rotation.x += 0.018;
            wireMesh.rotation.y += 0.008;

            // Hospital Crosses Slow Rotation
            hospitalNodes.forEach(h => {{
                h.crossH.rotation.y += 0.02;
                h.crossV.rotation.y += 0.02;
            }});

            if (isPlaying) {{
                phaseTimer += 1;
                if (phaseTimer > 240) {{
                    nextPhase();
                }}

                // Particle Motion Handling
                const positions = particleGeo.attributes.position.array;

                for (let i = 0; i < particleCount; i++) {{
                    const cIdx = pCurveIdx[i];
                    const cData = connectionCurves[cIdx];

                    if (!cData.active) {{
                        positions[i*3] = 9999;
                        continue;
                    }}

                    // Motion Direction based on Phase
                    if (phase === 1) {{
                        // Upload Phase: Move Downward from Hospital -> Server
                        pProgress[i] += pSpeed[i];
                        if (pProgress[i] > 1) pProgress[i] = 0;
                    }} else if (phase === 3) {{
                        // Broadcast Phase: Move Upward from Server -> Hospital
                        pProgress[i] -= pSpeed[i];
                        if (pProgress[i] < 0) pProgress[i] = 1;
                    }} else {{
                        // Mild idle oscillation
                        pProgress[i] += pSpeed[i] * 0.4;
                        if (pProgress[i] > 1) pProgress[i] = 0;
                    }}

                    const pt = cData.curve.getPoint(pProgress[i]);
                    positions[i*3] = pt.x;
                    positions[i*3+1] = pt.y;
                    positions[i*3+2] = pt.z;
                }}
                particleGeo.attributes.position.needsUpdate = true;

                // Aggregation Phase: Shockwave Pulse Effect
                if (phase === 2) {{
                    shockwaveScale += 0.12;
                    if (shockwaveScale > 14) shockwaveScale = 1;
                    shockwave.scale.set(shockwaveScale, shockwaveScale, 1);
                    shockwaveMat.opacity = Math.max(0, 0.8 - (shockwaveScale / 14));
                    coreMesh.material.emissiveIntensity = 1.0 + Math.sin(phaseTimer * 0.1) * 0.8;
                }} else {{
                    shockwaveMat.opacity = 0.0;
                    coreMesh.material.emissiveIntensity = 0.6;
                }}
            }}

            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / height;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, height);
        }});
    </script>
</body>
</html>
"""
    return html_code
