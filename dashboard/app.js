// Application state
let currentCondition = 1;
let socket = null;
let running = false;
let canvas, ctx;
let waitingChart, queueChart;

// Chart setups
const maxHistoryLength = 50;

// Setup Canvas and Charts on Load
window.onload = function() {
    setupCanvas();
    setupCharts();
    selectCondition(1);
};

function setupCanvas() {
    canvas = document.getElementById("map-canvas");
    ctx = canvas.getContext("2d");
    
    // Resize handler
    function resizeCanvas() {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        drawEmptyMap();
    }
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
}

function setupCharts() {
    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
            x: { display: false },
            y: {
                grid: { color: "rgba(255, 255, 255, 0.05)" },
                ticks: { color: "#9ca3af", font: { size: 9 } }
            }
        },
        animation: { duration: 0 }
    };

    const waitCtx = document.getElementById("waiting-chart").getContext("2d");
    waitingChart = new Chart(waitCtx, {
        type: 'line',
        data: {
            labels: Array(maxHistoryLength).fill(''),
            datasets: [{
                data: Array(maxHistoryLength).fill(0),
                borderColor: '#3b82f6',
                borderWidth: 1.5,
                fill: false,
                tension: 0.2,
                pointRadius: 0
            }]
        },
        options: chartOptions
    });

    const queueCtx = document.getElementById("queue-chart").getContext("2d");
    queueChart = new Chart(queueCtx, {
        type: 'line',
        data: {
            labels: Array(maxHistoryLength).fill(''),
            datasets: [{
                data: Array(maxHistoryLength).fill(0),
                borderColor: '#10b981',
                borderWidth: 1.5,
                fill: false,
                tension: 0.2,
                pointRadius: 0
            }]
        },
        options: chartOptions
    });
}

function selectCondition(condition) {
    if (running) return;
    currentCondition = condition;
    
    // Update active UI card
    document.querySelectorAll(".mode-card").forEach(card => {
        card.classList.remove("active");
    });
    const activeCard = document.querySelector(`.mode-card[data-condition="${condition}"]`);
    if (activeCard) {
        activeCard.classList.add("active");
    }
}

// REST actions
async function startSimulation() {
    if (running) return;
    
    // Update button states
    document.getElementById("btn-start").classList.add("disabled");
    document.getElementById("btn-start").disabled = true;
    document.querySelectorAll(".mode-card").forEach(c => c.disabled = true);
    
    const response = await fetch("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ condition: currentCondition })
    });
    
    const res = await response.json();
    if (res.status === "success") {
        running = true;
        document.getElementById("btn-stop").classList.remove("disabled");
        document.getElementById("btn-stop").disabled = false;
        document.getElementById("btn-inject").classList.remove("disabled");
        document.getElementById("btn-inject").disabled = false;
        
        document.getElementById("term-status").innerText = "Running";
        document.getElementById("term-status").style.color = "#10b981";
        
        connectWebSocket();
    } else {
        alert("Failed to start simulation.");
        resetUI();
    }
}

async function stopSimulation() {
    if (!running) return;
    
    await fetch("/api/stop", { method: "POST" });
    closeWebSocket();
    resetUI();
}

async function injectAmbulance() {
    if (!running) return;
    
    const response = await fetch("/api/inject", { method: "POST" });
    const res = await response.json();
    if (res.status === "success") {
        document.getElementById("btn-inject").classList.add("disabled");
        document.getElementById("btn-inject").disabled = true;
    }
}

function resetUI() {
    running = false;
    document.getElementById("btn-start").classList.remove("disabled");
    document.getElementById("btn-start").disabled = false;
    document.getElementById("btn-stop").classList.add("disabled");
    document.getElementById("btn-stop").disabled = true;
    document.getElementById("btn-inject").classList.add("disabled");
    document.getElementById("btn-inject").disabled = true;
    document.querySelectorAll(".mode-card").forEach(c => c.disabled = false);
    
    document.getElementById("term-status").innerText = "Idle";
    document.getElementById("term-status").style.color = "#9ca3af";
    
    // Reset KPIs
    document.getElementById("val-time").innerText = "0.0s";
    document.getElementById("val-wait").innerText = "0.0s";
    document.getElementById("val-queue").innerText = "0";
    
    document.getElementById("val-amb-status").innerText = "Inactive";
    document.getElementById("kpi-amb").classList.remove("active");
    document.getElementById("amb-substats").style.display = "none";
    
    drawEmptyMap();
}

// Websocket updates
function connectWebSocket() {
    const loc = window.location;
    const protocol = loc.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${loc.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        if (data.terminated) {
            stopSimulation();
            return;
        }
        if (data.time !== undefined) {
            updateDashboard(data);
        }
    };
    
    socket.onclose = function() {
        if (running) resetUI();
    };
}

function closeWebSocket() {
    if (socket) {
        socket.close();
        socket = null;
    }
}

// Logs terminal writer
function appendLog(text) {
    const terminal = document.getElementById("terminal-logs");
    const div = document.createElement("div");
    
    if (text.includes("🚨") || text.includes("[PRIORITY]")) {
        div.className = "log-line priority";
    } else if (text.includes("complete") || text.includes("released")) {
        div.className = "log-line success";
    } else if (text.includes("Loaded") || text.includes("initialised")) {
        div.className = "log-line info";
    } else {
        div.className = "log-line system";
    }
    
    div.innerText = text;
    terminal.appendChild(div);
    terminal.scrollTop = terminal.scrollHeight;
}

function updateDashboard(data) {
    // 1. Update KPI Text
    document.getElementById("val-time").innerText = `${data.time.toFixed(1)}s`;
    document.getElementById("val-wait").innerText = `${data.metrics.waiting_time.toFixed(1)}s`;
    document.getElementById("val-queue").innerText = Math.round(data.metrics.queue_length);
    
    // 2. Update Logs Terminal
    if (data.logs && data.logs.length > 0) {
        data.logs.forEach(msg => appendLog(msg));
    }
    
    // 3. Ambulance KPIs
    const kpiAmb = document.getElementById("kpi-amb");
    const ambStatus = document.getElementById("val-amb-status");
    const ambSub = document.getElementById("amb-substats");
    
    if (data.ambulance.injected) {
        kpiAmb.classList.add("active");
        if (data.ambulance.active) {
            ambStatus.innerText = "Approaching";
            ambStatus.style.color = "#ef4444";
            ambSub.style.display = "flex";
            document.getElementById("val-amb-time").innerText = `${(data.time - data.ambulance.start_time).toFixed(1)}s`;
            document.getElementById("val-amb-wait").innerText = `${data.ambulance.waiting_time.toFixed(1)}s`;
        } else if (data.ambulance.travel_time !== null) {
            ambStatus.innerText = "Completed";
            ambStatus.style.color = "#10b981";
            ambSub.style.display = "flex";
            document.getElementById("val-amb-time").innerText = `${data.ambulance.travel_time.toFixed(1)}s`;
            document.getElementById("val-amb-wait").innerText = `${data.ambulance.waiting_time.toFixed(1)}s`;
        }
    } else {
        kpiAmb.classList.remove("active");
        ambStatus.innerText = "Inactive";
        ambStatus.style.color = "";
        ambSub.style.display = "none";
    }
    
    // 4. Update Charts
    updateChart(waitingChart, data.metrics.waiting_time_history);
    updateChart(queueChart, data.metrics.queue_length_history);
    
    // 5. Draw Visual Canvas
    drawMap(data);
}

function updateChart(chart, history) {
    const padded = [...history];
    while (padded.length < maxHistoryLength) {
        padded.unshift(0);
    }
    if (padded.length > maxHistoryLength) {
        padded.splice(0, padded.length - maxHistoryLength);
    }
    chart.data.datasets[0].data = padded;
    chart.update();
}

// Visual Map Drawing Functions
function drawEmptyMap() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = "#1e293b";
    ctx.font = "14px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Simulation offline. Click 'Start Simulation'.", canvas.width / 2, canvas.height / 2);
}

function drawMap(data) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const center = data.tls.center;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    
    // Coordinate scale: 1 meter = X pixels.
    // Let's dynamically fit 180 meters around the junction
    const viewRadiusMeters = 180.0;
    const scale = Math.min(canvas.width, canvas.height) / (viewRadiusMeters * 2);
    
    // Draw Roads (Grid layout around junction center)
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = scale * 16.0; // 16m wide road
    
    // Vertical road
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, canvas.height);
    ctx.stroke();
    
    // Horizontal road
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(canvas.width, cy);
    ctx.stroke();
    
    // Draw Lane Markings (Dotted lines)
    ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    
    // Vertical lane divider
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, canvas.height);
    ctx.stroke();
    
    // Horizontal lane divider
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(canvas.width, cy);
    ctx.stroke();
    
    ctx.setLineDash([]); // Reset
    
    // Draw TLS Junction highlight circle
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.fillStyle = "rgba(255, 255, 255, 0.02)";
    ctx.beginPath();
    ctx.arc(cx, cy, scale * 30.0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Draw Proximity sensor radius in MAESTRO-FL mode
    if (data.condition === 4) {
        ctx.strokeStyle = "rgba(239, 68, 68, 0.15)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.arc(cx, cy, scale * 100.0, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.setLineDash([]);
        
        ctx.fillStyle = "rgba(239, 68, 68, 0.03)";
        ctx.fill();
    }
    
    // Draw Traffic Lights (North, South, East, West)
    const tlsState = data.tls.state;
    // Map TLS characters to compass directions
    // OSM fixed net representation controls 4 incoming lanes:
    // Green phase mappings can be dynamically shown on roads:
    drawTrafficLightBulb(cx, cy - scale * 24.0, tlsState[0] || 'r'); // North approach
    drawTrafficLightBulb(cx, cy + scale * 24.0, tlsState[2] || 'r'); // South approach
    drawTrafficLightBulb(cx + scale * 24.0, cy, tlsState[1] || 'r'); // East approach
    drawTrafficLightBulb(cx - scale * 24.0, cy, tlsState[3] || 'r'); // West approach

    // Draw Vehicles
    if (data.vehicles && data.vehicles.length > 0) {
        data.vehicles.forEach(veh => {
            // Translate relative to center
            const rx = veh.x - center[0];
            const ry = -(veh.y - center[1]); // SUMO is Y-up, Canvas is Y-down
            
            const vx = cx + rx * scale;
            const vy = cy + ry * scale;
            
            // Skip drawing if outside canvas
            if (vx < 0 || vx > canvas.width || vy < 0 || vy > canvas.height) return;
            
            if (veh.type === "emergency") {
                // Flash animation for ambulance
                const flash = Math.floor(Date.now() / 150) % 2 === 0;
                
                // Draw ambulance glow
                ctx.fillStyle = flash ? "rgba(239, 68, 68, 0.6)" : "rgba(59, 130, 246, 0.6)";
                ctx.beginPath();
                ctx.arc(vx, vy, scale * 6.5, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw ambulance body
                ctx.fillStyle = "#ef4444";
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(vx, vy, scale * 4.0, 0, 2 * Math.PI);
                ctx.fill();
                ctx.stroke();
                
                // Direction indicator
                ctx.fillStyle = "#ffffff";
                ctx.beginPath();
                const rad = -veh.angle * Math.PI / 180;
                ctx.arc(vx + Math.sin(rad) * scale * 3.5, vy - Math.cos(rad) * scale * 3.5, scale * 1.2, 0, 2*Math.PI);
                ctx.fill();
            } else {
                // Normal vehicle
                ctx.fillStyle = "rgba(96, 165, 250, 0.8)";
                ctx.beginPath();
                ctx.arc(vx, vy, scale * 3.0, 0, 2 * Math.PI);
                ctx.fill();
                
                // Small heading indicator
                ctx.fillStyle = "#ffffff";
                ctx.beginPath();
                const rad = -veh.angle * Math.PI / 180;
                ctx.arc(vx + Math.sin(rad) * scale * 2.5, vy - Math.cos(rad) * scale * 2.5, scale * 0.8, 0, 2*Math.PI);
                ctx.fill();
            }
        });
    }
}

function drawTrafficLightBulb(x, y, charState) {
    let color = "#ef4444"; // red
    let glow = "rgba(239, 68, 68, 0.4)";
    if (charState.toLowerCase() === 'g') {
        color = "#10b981"; // green
        glow = "rgba(16, 185, 129, 0.4)";
    } else if (charState.toLowerCase() === 'y' || charState.toLowerCase() === 'u') {
        color = "#f59e0b"; // yellow
        glow = "rgba(245, 158, 11, 0.4)";
    }
    
    // Draw housing
    ctx.fillStyle = "#1e293b";
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, 2*Math.PI);
    ctx.fill();
    
    // Draw active light bulb
    ctx.fillStyle = color;
    ctx.shadowBlur = 8;
    ctx.shadowColor = glow;
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, 2*Math.PI);
    ctx.fill();
    ctx.shadowBlur = 0; // reset
}
