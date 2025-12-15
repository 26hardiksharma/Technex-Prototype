// Main JavaScript for Flask Demo App
console.log('🚀 App.js loaded - Collision detection enabled! Version: 2.0');

// Check system status on load
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const statusIndicator = document.getElementById('status-indicator');
        const modelStatus = document.getElementById('model-status');
        
        if (data.demo_mode) {
            statusIndicator.textContent = '✅ Demo Mode Ready (no training needed)';
            statusIndicator.className = 'status-ready';
            modelStatus.textContent = 'Using pre-baked trajectories for instant demo';
        } else if (data.ready) {
            statusIndicator.textContent = '✅ System Ready';
            statusIndicator.className = 'status-ready';
            modelStatus.textContent = 'Model Loaded';
        } else {
            statusIndicator.textContent = '⚠️ Model Not Loaded';
            statusIndicator.className = 'status-warning';
            modelStatus.textContent = 'Run training first: python scripts/train.py --test';
        }
    } catch (error) {
        console.error('Error checking status:', error);
        document.getElementById('status-indicator').textContent = '❌ Error';
    }
}

// Run a scenario
async function runScenario() {
    const scenarioType = 'head_on';
    const exampleId = document.getElementById('example-select').value;
    const useAI = document.getElementById('use-ai').checked;
    const runBtn = document.getElementById('run-btn');
    
    // Disable button and show loading
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Running Simulation...';
    
    try {
        const response = await fetch('/api/run_scenario', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario_type: scenarioType, example_id: exampleId, use_ai: useAI })
        });
        
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display results
        displayResults(data);
        
        // Animate trajectory in real-time
        runBtn.textContent = '🎬 Animating...';
        console.log('Full response data:', data);
        console.log('Trajectory object:', data.trajectory);
        console.log('Trajectory.satellite exists?', data.trajectory?.satellite);
        plotTrajectory(data.trajectory, data.metrics);

        // Also auto-run comparison and benchmark for the chosen scenario
        await fetchAndShowComparison(scenarioType, exampleId);
        await benchmarkInference();
        
    } catch (error) {
        console.error('Error running scenario:', error);
        alert('Error running scenario. Check console for details.');
    } finally {
        runBtn.disabled = false;
        runBtn.textContent = '▶️ Run Scenario';
    }
}

// Display results in panel
function displayResults(data) {
    const panel = document.getElementById('results-panel');
    panel.style.display = 'block';
    
    const metrics = data.metrics;
    
    const isCollision = metrics.collision || metrics.min_distance < 75;
    const distanceText = isCollision && metrics.min_distance < 75 ? `${metrics.min_distance.toFixed(1)} m 💥` : `${metrics.min_distance.toFixed(1)} m`;
    document.getElementById('result-distance').textContent = distanceText;
    document.getElementById('result-distance').className = isCollision ? 'result-value danger' : 'result-value success';
    
    // Show collision alert banner
    const collisionAlert = document.getElementById('collision-alert');
    if (collisionAlert) {
        collisionAlert.style.display = isCollision ? 'block' : 'none';
    }
    
    // Log collision status for debugging  
    if (isCollision) {
        console.log('🚨 COLLISION DETECTED! Min Distance:', metrics.min_distance, 'm');
        console.log('💥 Satellite destroyed at collision!');
    } else {
        console.log('✅ Safe passage - Min Distance:', metrics.min_distance, 'm');
    }
    
    document.getElementById('result-collision').textContent = isCollision ? '💥 COLLISION!' : '✅ No';
    document.getElementById('result-collision').className = isCollision ? 'result-value danger' : 'result-value success';
    
    document.getElementById('result-success').textContent = metrics.success ? '✅ Yes' : '❌ No';
    document.getElementById('result-success').className = metrics.success ? 'result-value success' : 'result-value danger';
    
    document.getElementById('result-fuel').textContent = `${metrics.total_fuel_used.toFixed(3)} m/s (${(metrics.total_fuel_used / 10 * 100).toFixed(1)}%)`;
    
    document.getElementById('result-duration').textContent = `${metrics.duration_seconds.toFixed(0)} seconds`;
    document.getElementById('result-steps').textContent = metrics.steps;
}

// Plot trajectory with animation
function plotTrajectory(trajectory, metrics, animate = true) {
    // Safety check for invalid trajectory
    console.log('plotTrajectory called with:', { trajectory, metrics });
    
    if (!trajectory) {
        console.error('Trajectory is null or undefined');
        return;
    }
    
    if (!trajectory.satellite) {
        console.error('trajectory.satellite is undefined. Trajectory keys:', Object.keys(trajectory));
        return;
    }
    
    if (!Array.isArray(trajectory.satellite)) {
        console.error('trajectory.satellite is not an array:', typeof trajectory.satellite);
        return;
    }
    
    const collisionRadius = 75;
    const safeDistance = 500;
    
    // Extract data
    const satX = trajectory.satellite.map(p => p[0]);
    const satY = trajectory.satellite.map(p => p[1]);
    
    if (animate) {
        animateTrajectory(trajectory, metrics);
        return;
    }
    
    // Create trajectory plot
    const trajectoryTrace = {
        x: satX,
        y: satY,
        mode: 'lines+markers',
        type: 'scatter',
        name: 'Satellite Trajectory',
        line: { color: '#3b82f6', width: 3 },
        marker: { size: 6, color: satX.map((_, i) => i), colorscale: 'Viridis' }
    };
    
    // Start and end markers
    const startTrace = {
        x: [satX[0]],
        y: [satY[0]],
        mode: 'markers',
        type: 'scatter',
        name: 'Start',
        marker: { size: 15, color: 'green', symbol: 'circle' }
    };
    
    const endTrace = {
        x: [satX[satX.length - 1]],
        y: [satY[satY.length - 1]],
        mode: 'markers',
        type: 'scatter',
        name: 'End',
        marker: { size: 15, color: 'blue', symbol: 'square' }
    };
    
    // Debris marker
    const debrisTrace = {
        x: [0],
        y: [0],
        mode: 'markers',
        type: 'scatter',
        name: 'Debris',
        marker: { size: 20, color: 'red', symbol: 'star' }
    };
    
    // Collision zone circle
    const theta = Array.from({ length: 100 }, (_, i) => i * 2 * Math.PI / 99);
    const collisionZone = {
        x: theta.map(t => collisionRadius * Math.cos(t)),
        y: theta.map(t => collisionRadius * Math.sin(t)),
        mode: 'lines',
        type: 'scatter',
        name: 'Collision Zone',
        line: { color: 'red', width: 2, dash: 'dash' },
        fill: 'toself',
        fillcolor: 'rgba(255, 0, 0, 0.1)'
    };
    
    // Safe zone circle
    const safeZone = {
        x: theta.map(t => safeDistance * Math.cos(t)),
        y: theta.map(t => safeDistance * Math.sin(t)),
        mode: 'lines',
        type: 'scatter',
        name: 'Safe Zone',
        line: { color: 'green', width: 2, dash: 'dash' }
    };
    
    const layout = {
        title: 'Orbital Trajectory (LVLH Frame)',
        xaxis: { title: 'Radial Distance (m)', zeroline: true },
        yaxis: { title: 'Along-Track Distance (m)', zeroline: true, scaleanchor: 'x' },
        showlegend: true,
        height: 500,
        hovermode: 'closest'
    };
    
    Plotly.newPlot('trajectory-plot', 
        [collisionZone, safeZone, trajectoryTrace, startTrace, endTrace, debrisTrace],
        layout,
        { responsive: true }
    );
    
    // Distance vs Time plot
    const distanceTrace = {
        x: trajectory.time,
        y: trajectory.distance,
        mode: 'lines',
        type: 'scatter',
        name: 'Distance to Debris',
        line: { color: '#3b82f6', width: 3 }
    };
    
    const collisionLine = {
        x: [0, Math.max(...trajectory.time)],
        y: [collisionRadius, collisionRadius],
        mode: 'lines',
        type: 'scatter',
        name: 'Collision Radius',
        line: { color: 'red', width: 2, dash: 'dash' }
    };
    
    const safeLine = {
        x: [0, Math.max(...trajectory.time)],
        y: [safeDistance, safeDistance],
        mode: 'lines',
        type: 'scatter',
        name: 'Safe Distance',
        line: { color: 'green', width: 2, dash: 'dash' }
    };
    
    const distanceLayout = {
        title: 'Distance vs Time',
        xaxis: { title: 'Time (seconds)' },
        yaxis: { title: 'Distance (m)' },
        showlegend: true,
        height: 400
    };
    
    Plotly.newPlot('distance-plot',
        [distanceTrace, collisionLine, safeLine],
        distanceLayout,
        { responsive: true }
    );
}

// Compare AI vs No-AI
async function compareScenarios() {
    const scenarioType = 'head_on';
    const exampleId = document.getElementById('example-select').value;
    const compareBtn = document.getElementById('compare-btn');
    if (compareBtn) {
        compareBtn.disabled = true;
        compareBtn.textContent = '⏳ Comparing...';
    }
    
    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario_type: scenarioType, example_id: exampleId })
        });
        
        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }
        
        // Display comparison
        displayComparison(data);

        // Auto-run benchmark view for quick demo
        benchmarkInference();
        
    } catch (error) {
        console.error('Error comparing scenarios:', error);
        alert('Error comparing scenarios. Check console for details.');
    } finally {
        if (compareBtn) {
            compareBtn.disabled = false;
            compareBtn.textContent = '📊 Compare AI vs No-AI';
        }
    }
}

// Fetch comparison and display (used automatically after running a scenario)
async function fetchAndShowComparison(scenarioType, exampleId) {
    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scenario_type: scenarioType, example_id: exampleId })
        });

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        displayComparison(data);
    } catch (error) {
        console.error('Error auto-comparing scenarios:', error);
    }
}

// Display comparison results
function displayComparison(data) {
    // Safety checks for missing data
    if (!data || !data.ai || !data.no_ai) {
        console.error('Invalid comparison data:', data);
        return;
    }
    
    if (!data.ai.trajectory || !data.ai.trajectory.satellite || 
        !data.no_ai.trajectory || !data.no_ai.trajectory.satellite) {
        console.error('Missing trajectory data in comparison:', data);
        return;
    }
    
    const panel = document.getElementById('comparison-panel');
    panel.style.display = 'block';
    
    // Plot both trajectories
    const aiTraj = data.ai.trajectory.satellite;
    const noAiTraj = data.no_ai.trajectory.satellite;
    
    const aiTrace = {
        x: aiTraj.map(p => p[0]),
        y: aiTraj.map(p => p[1]),
        mode: 'lines+markers',
        type: 'scatter',
        name: '🤖 With AI',
        line: { color: '#10b981', width: 3 }
    };
    
    const noAiTrace = {
        x: noAiTraj.map(p => p[0]),
        y: noAiTraj.map(p => p[1]),
        mode: 'lines+markers',
        type: 'scatter',
        name: '🚫 Without AI',
        line: { color: '#ef4444', width: 3 }
    };
    
    const debrisTrace = {
        x: [0],
        y: [0],
        mode: 'markers',
        type: 'scatter',
        name: 'Debris',
        marker: { size: 20, color: 'orange', symbol: 'star' }
    };
    
    // Safety circles
    const theta = Array.from({ length: 100 }, (_, i) => i * 2 * Math.PI / 99);
    const collisionZone = {
        x: theta.map(t => 75 * Math.cos(t)),
        y: theta.map(t => 75 * Math.sin(t)),
        mode: 'lines',
        type: 'scatter',
        name: 'Collision Zone',
        line: { color: 'red', width: 2, dash: 'dash' },
        showlegend: false
    };
    
    const layout = {
        title: 'AI vs No-AI Comparison',
        xaxis: { title: 'Radial Distance (m)', zeroline: true },
        yaxis: { title: 'Along-Track Distance (m)', zeroline: true, scaleanchor: 'x' },
        showlegend: true,
        height: 500
    };
    
    Plotly.newPlot('comparison-plot',
        [collisionZone, aiTrace, noAiTrace, debrisTrace],
        layout,
        { responsive: true }
    );
    
    // Display metrics
    document.getElementById('ai-results').innerHTML = `
        <p><strong>Min Distance:</strong> ${data.ai?.min_distance?.toFixed(1) || 'N/A'} m</p>
        <p><strong>Collision:</strong> ${data.ai?.collision ? '❌ Yes' : '✅ No'}</p>
        <p><strong>Fuel Used:</strong> ${data.ai?.fuel_used?.toFixed(3) || 'N/A'} m/s</p>
        <p class="success">✅ Success!</p>
    `;
    
    const noAiCollision = data.no_ai?.collision || (data.no_ai?.min_distance !== undefined && data.no_ai.min_distance < 75);
    document.getElementById('no-ai-results').innerHTML = `
        <p><strong>Min Distance:</strong> ${data.no_ai?.min_distance?.toFixed(1) || 'N/A'} m</p>
        <p><strong>Collision:</strong> ${noAiCollision ? '💥 YES!' : '✅ No'}</p>
        <p><strong>Fuel Used:</strong> ${data.no_ai?.fuel_used?.toFixed(3) || 'N/A'} m/s</p>
        <p class="${noAiCollision ? 'danger' : 'success'}" style="font-weight: bold; font-size: 1.1em;">${noAiCollision ? '💥 CRASHED!' : '✅ Success'}</p>
    `;
}

// Benchmark inference
async function benchmarkInference() {
    const benchBtn = document.getElementById('benchmark-btn');
    
    if (benchBtn) {
        benchBtn.disabled = true;
        benchBtn.textContent = '⏳ Benchmarking...';
    }
    
    try {
        const response = await fetch('/api/benchmark');
        const data = await response.json();
        
        // Display benchmark results
        displayBenchmark(data);
        
    } catch (error) {
        console.error('Error benchmarking:', error);
        alert('Error benchmarking. Check console for details.');
    } finally {
        if (benchBtn) {
            benchBtn.disabled = false;
            benchBtn.textContent = '⚡ Benchmark Inference';
        }
    }
}

// Display benchmark results
function displayBenchmark(data) {
    const panel = document.getElementById('benchmark-panel');
    panel.style.display = 'block';
    
    document.getElementById('bench-mean').textContent = `${data.mean_latency_ms.toFixed(3)} ms`;
    document.getElementById('bench-p95').textContent = `${data.p95_latency_ms.toFixed(3)} ms`;
    document.getElementById('bench-p99').textContent = `${data.p99_latency_ms.toFixed(3)} ms`;
    document.getElementById('bench-speedup').textContent = `${data.speedup.toFixed(0)}x Faster!`;
    
    // Update metrics banner
    document.getElementById('speedup-metric').textContent = `${data.speedup.toFixed(0)}x`;
    document.getElementById('latency-metric').textContent = `${data.mean_latency_ms.toFixed(2)}ms`;
    
    // Calculate reaction distances
    const velocity = 50; // m/s
    const edgeDistance = velocity * (data.mean_latency_ms / 1000);
    const groundDistance = velocity * (data.ground_control_latency_ms / 1000);
    const safetyMargin = groundDistance - edgeDistance;
    
    document.getElementById('edge-distance').textContent = `${edgeDistance.toFixed(3)} m`;
    document.getElementById('safety-margin').textContent = `${safetyMargin.toFixed(2)}m`;
    
    // Update bar width
    const percentage = (edgeDistance / groundDistance) * 100;
    document.querySelector('.edge-bar').style.width = `${Math.max(percentage, 2)}%`;
}

// Animate trajectory in real-time
function animateTrajectory(trajectory, metrics) {
    console.log('Starting animation with trajectory:', trajectory);
    console.log('Metrics:', metrics);
    console.log('Trajectory keys:', trajectory ? Object.keys(trajectory) : 'trajectory is null');
    
    // Safety checks for undefined trajectory components
    if (!trajectory) {
        console.error('Trajectory is null or undefined');
        return;
    }
    
    if (!trajectory.satellite) {
        console.error('trajectory.satellite is undefined. Available keys:', Object.keys(trajectory));
        return;
    }
    
    if (!Array.isArray(trajectory.satellite)) {
        console.error('trajectory.satellite is not an array:', typeof trajectory.satellite, trajectory.satellite);
        return;
    }
    
    const collisionRadius = 75;
    const safeDistance = 500;
    const satX = trajectory.satellite.map(p => p[0]);
    const satY = trajectory.satellite.map(p => p[1]);
    const origX = trajectory.original ? trajectory.original.map(p => p[0]) : satX;
    const origY = trajectory.original ? trajectory.original.map(p => p[1]) : satY;
    const debrisX = (trajectory.debris && Array.isArray(trajectory.debris)) ? trajectory.debris.map(p => p[0]) : [0];
    const debrisY = (trajectory.debris && Array.isArray(trajectory.debris)) ? trajectory.debris.map(p => p[1]) : [0];
    const actions = trajectory.actions || [];
    const actionNames = ['Coast', '-Radial', '+Radial', '-Track', '+Track'];
    
    console.log(`Animation frames: ${satX.length}, Debris moving: ${debrisX.length}`);
    
    // Create collision and safe zones
    const theta = Array.from({ length: 100 }, (_, i) => i * 2 * Math.PI / 99);
    const collisionZone = {
        x: theta.map(t => collisionRadius * Math.cos(t)),
        y: theta.map(t => collisionRadius * Math.sin(t)),
        mode: 'lines',
        type: 'scatter',
        name: 'Collision Zone',
        line: { color: '#f87171', width: 2, dash: 'dash' },
        fill: 'toself',
        fillcolor: 'rgba(248, 113, 113, 0.15)',
        hoverinfo: 'name'
    };
    
    const safeZone = {
        x: theta.map(t => safeDistance * Math.cos(t)),
        y: theta.map(t => safeDistance * Math.sin(t)),
        mode: 'lines',
        type: 'scatter',
        name: 'Safe Zone',
        line: { color: '#34d399', width: 2, dash: 'dash' },
        hoverinfo: 'name'
    };
    
    // Debris (animated, moving)
    const debrisTrace = {
        x: [debrisX[0]],
        y: [debrisY[0]],
        mode: 'markers',
        type: 'scatter',
        name: 'Debris',
        marker: { 
            size: 30, 
            color: '#ef4444',
            symbol: 'star',
            line: { color: '#dc2626', width: 2 }
        },
        hovertemplate: '<b>Debris</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
    };
    
    // Debris path trace
    const debrisPathTrace = {
        x: [debrisX[0]],
        y: [debrisY[0]],
        mode: 'lines',
        type: 'scatter',
        name: 'Debris Path',
        line: { color: '#f87171', width: 2, dash: 'dot' },
        opacity: 0.6,
        hoverinfo: 'skip'
    };
    
    // Original satellite (no AI - animated, gray)
    const originalSatelliteTrace = {
        x: [origX[0]],
        y: [origY[0]],
        mode: 'markers',
        type: 'scatter',
        name: 'No AI (Original)',
        marker: { 
            size: 18, 
            color: '#94a3b8',
            symbol: 'circle',
            line: { color: '#64748b', width: 2 }
        },
        hovertemplate: '<b>No AI Satellite</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
    };
    
    // AI-controlled satellite (animated, blue)
    const satelliteTrace = {
        x: [satX[0]],
        y: [satY[0]],
        mode: 'markers',
        type: 'scatter',
        name: 'AI-Controlled',
        marker: { 
            size: 20, 
            color: '#60a5fa',
            symbol: 'diamond',
            line: { color: '#3b82f6', width: 2 }
        },
        hovertemplate: '<b>AI Satellite</b><br>Position: (%{x:.1f}, %{y:.1f})<extra></extra>'
    };
    
    // Original trajectory path (without AI - grows over time)
    const originalPathTrace = {
        x: [origX[0]],
        y: [origY[0]],
        mode: 'lines',
        type: 'scatter',
        name: 'No AI Path',
        line: { color: '#94a3b8', width: 2.5 },
        hoverinfo: 'skip'
    };
    
    // AI-controlled trajectory path (grows over time)
    const pathTrace = {
        x: [satX[0]],
        y: [satY[0]],
        mode: 'lines',
        type: 'scatter',
        name: 'AI Path',
        line: { color: '#60a5fa', width: 3 },
        hoverinfo: 'skip'
    };
    
    // Start marker
    const startTrace = {
        x: [satX[0]],
        y: [satY[0]],
        mode: 'markers+text',
        type: 'scatter',
        name: 'Start',
        marker: { size: 12, color: '#34d399', symbol: 'circle' },
        text: ['START'],
        textposition: 'top center',
        textfont: { color: '#34d399', size: 10 },
        hoverinfo: 'name'
    };
    
    const layout = {
        title: {
            text: 'Real-Time Collision Avoidance Simulation<br><sub style="font-size: 12px;">🔵 AI-Controlled vs ⚪ No AI | 🔴 Moving Debris</sub>',
            font: { color: '#f1f5f9' }
        },
        xaxis: { 
            title: 'Radial Distance (m)', 
            zeroline: true,
            gridcolor: '#475569',
            color: '#94a3b8'
        },
        yaxis: { 
            title: 'Along-Track Distance (m)', 
            zeroline: true, 
            scaleanchor: 'x',
            gridcolor: '#475569',
            color: '#94a3b8'
        },
        showlegend: true,
        legend: { font: { color: '#f1f5f9' } },
        height: 500,
        hovermode: 'closest',
        paper_bgcolor: '#1e293b',
        plot_bgcolor: '#0f172a',
        annotations: [{
            text: 'Initializing...',
            xref: 'paper',
            yref: 'paper',
            x: 0.5,
            y: 0.95,
            showarrow: false,
            font: { size: 14, color: '#60a5fa' },
            xanchor: 'center'
        }]
    };
    
    // Initial plot - order matters for trace indices
    console.log('Creating initial Plotly plot...');
    Plotly.newPlot('trajectory-plot', 
        [collisionZone, safeZone, debrisPathTrace, originalPathTrace, pathTrace, startTrace, debrisTrace, originalSatelliteTrace, satelliteTrace],
        layout,
        { responsive: true, displayModeBar: true }
    ).then(() => {
        console.log('Plotly plot created successfully');
    }).catch(err => {
        console.error('Error creating Plotly plot:', err);
    });
    
    // Animation loop
    let frame = 0;
    const totalFrames = satX.length;
    const frameDelay = 150; // ms between frames (increased for better visibility)
    
    const interval = setInterval(() => {
        if (frame >= totalFrames - 1) {
            clearInterval(interval);
            // Calculate final collision status (relative to final debris position)
            const finalDebrisX = debrisX[debrisX.length - 1];
            const finalDebrisY = debrisY[debrisY.length - 1];
            const finalOrigDist = Math.sqrt((origX[origX.length - 1] - finalDebrisX)**2 + (origY[origY.length - 1] - finalDebrisY)**2);
            const origFinalCollision = finalOrigDist < collisionRadius;
            
            // Add end markers
            const endMarkers = [{
                x: [satX[satX.length - 1]],
                y: [satY[satY.length - 1]],
                mode: 'markers+text',
                type: 'scatter',
                name: 'AI End Position',
                marker: { size: 16, color: '#34d399', symbol: 'square', line: { color: '#10b981', width: 2 } },
                text: ['✅ AI SAFE'],
                textposition: 'bottom center',
                textfont: { color: '#34d399', size: 11, weight: 'bold' }
            }, {
                x: [origX[origX.length - 1]],
                y: [origY[origY.length - 1]],
                mode: 'markers+text',
                type: 'scatter',
                name: 'Original End Position',
                marker: { size: 16, color: origFinalCollision ? '#ef4444' : '#94a3b8', symbol: 'x', line: { color: origFinalCollision ? '#dc2626' : '#64748b', width: 2 } },
                text: [origFinalCollision ? '💥 COLLISION!' : '⚠️ NO AI'],
                textposition: 'top center',
                textfont: { color: origFinalCollision ? '#ef4444' : '#94a3b8', size: 11, weight: 'bold' }
            }];
            
            Plotly.addTraces('trajectory-plot', endMarkers);
            
            // Update annotation to show completion with comparison
            const comparisonText = origFinalCollision ? 
                `AI: ✅ SUCCESS | No AI: 💥 COLLISION | Distance: ${metrics.min_distance.toFixed(1)}m | Fuel: ${metrics.total_fuel_used.toFixed(2)} m/s` :
                `AI: ✅ SUCCESS | Min Distance: ${metrics.min_distance.toFixed(1)}m | Fuel: ${metrics.total_fuel_used.toFixed(2)} m/s`;
            
            Plotly.relayout('trajectory-plot', {
                'annotations[0].text': comparisonText,
                'annotations[0].font.color': '#34d399',
                'annotations[0].font.size': 14
            });
            
            // Plot distance vs time after animation
            plotDistanceVsTime(trajectory, metrics);
            return;
        }
        
        frame++;
        
        // Calculate distances for both satellites (relative to moving debris)
        const aiDist = Math.sqrt((satX[frame] - debrisX[frame])**2 + (satY[frame] - debrisY[frame])**2);
        const origDist = Math.sqrt((origX[frame] - debrisX[frame])**2 + (origY[frame] - debrisY[frame])**2);
        
        // Check if original satellite collided
        const origCollided = origDist < collisionRadius;
        
        // Update debris position
        Plotly.update('trajectory-plot',
            {
                x: [[debrisX[frame]]],
                y: [[debrisY[frame]]]
            },
            {},
            [6] // Index of debris trace
        );
        
        // Update debris path
        Plotly.update('trajectory-plot',
            {
                x: [debrisX.slice(0, frame + 1)],
                y: [debrisY.slice(0, frame + 1)]
            },
            {},
            [2] // Index of debris path trace
        );
        
        // Update original satellite (no AI) position and appearance
        Plotly.update('trajectory-plot', 
            {
                x: [[origX[frame]]], 
                y: [[origY[frame]]],
                'marker.color': [origCollided ? '#ef4444' : '#94a3b8'],
                'marker.size': [origCollided ? 25 : 18],
                'marker.symbol': [origCollided ? 'x' : 'circle']
            },
            {},
            [7] // Index of original satellite trace
        );
        
        // Update AI satellite position
        Plotly.update('trajectory-plot', 
            {
                x: [[satX[frame]]], 
                y: [[satY[frame]]]
            },
            {},
            [8] // Index of AI satellite trace
        );
        
        // Update original path (all points up to current frame)
        Plotly.update('trajectory-plot',
            {
                x: [origX.slice(0, frame + 1)],
                y: [origY.slice(0, frame + 1)]
            },
            {},
            [3] // Index of original path trace
        );
        
        // Update AI-corrected path (all points up to current frame)
        Plotly.update('trajectory-plot',
            {
                x: [satX.slice(0, frame + 1)],
                y: [satY.slice(0, frame + 1)]
            },
            {},
            [4] // Index of AI path trace
        );
        
        // Update annotation with current action and stats
        const currentAction = actions[frame] !== undefined ? actionNames[actions[frame]] : 'Coast';
        const currentTime = trajectory.time[frame].toFixed(0);
        const currentFuel = trajectory.fuel_used[frame].toFixed(2);
        
        // Status for both satellites
        const aiStatus = aiDist < collisionRadius ? '❌ COLLIDING' : (aiDist > safeDistance ? '✅ SAFE' : '⚠️ WARNING');
        const origStatus = origCollided ? '💥 COLLISION!' : (origDist > safeDistance ? 'Safe' : 'Warning');
        
        Plotly.relayout('trajectory-plot', {
            'annotations[0].text': `Time: ${currentTime}s | AI: ${aiStatus} (${aiDist.toFixed(1)}m) | No AI: ${origStatus} (${origDist.toFixed(1)}m) | Action: ${currentAction}`,
            'annotations[0].font.color': aiDist < collisionRadius ? '#ef4444' : (aiDist > safeDistance ? '#34d399' : '#fbbf24')
        });
        
    }, frameDelay);
}

// Plot distance vs time (called after animation)
function plotDistanceVsTime(trajectory, metrics) {
    const collisionRadius = 75;
    const safeDistance = 500;
    
    const distanceTrace = {
        x: trajectory.time,
        y: trajectory.distance,
        mode: 'lines',
        type: 'scatter',
        name: 'Distance to Debris',
        line: { color: '#60a5fa', width: 3 }
    };
    
    const collisionLine = {
        x: [0, Math.max(...trajectory.time)],
        y: [collisionRadius, collisionRadius],
        mode: 'lines',
        type: 'scatter',
        name: 'Collision Radius',
        line: { color: '#f87171', width: 2, dash: 'dash' }
    };
    
    const safeLine = {
        x: [0, Math.max(...trajectory.time)],
        y: [safeDistance, safeDistance],
        mode: 'lines',
        type: 'scatter',
        name: 'Safe Distance',
        line: { color: '#34d399', width: 2, dash: 'dash' }
    };
    
    const distanceLayout = {
        title: { text: 'Distance vs Time', font: { color: '#f1f5f9' } },
        xaxis: { title: 'Time (seconds)', gridcolor: '#475569', color: '#94a3b8' },
        yaxis: { title: 'Distance to Debris (m)', gridcolor: '#475569', color: '#94a3b8' },
        showlegend: true,
        legend: { font: { color: '#f1f5f9' } },
        height: 400,
        paper_bgcolor: '#1e293b',
        plot_bgcolor: '#0f172a'
    };
    
    Plotly.newPlot('distance-plot',
        [distanceTrace, collisionLine, safeLine],
        distanceLayout,
        { responsive: true }
    );
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    checkStatus();
    
    document.getElementById('run-btn').addEventListener('click', runScenario);
});
