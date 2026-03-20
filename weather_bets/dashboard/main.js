// Weather Bets Dashboard — spread betting UI
const API_URL = '/api/state';
const REFRESH_MS = 5000;

async function fetchState() {
    try {
        const resp = await fetch(API_URL);
        return await resp.json();
    } catch (e) {
        console.error('Fetch error:', e);
        return null;
    }
}

function renderForecasts(forecasts, bias) {
    const el = document.getElementById('forecast-cards');
    if (!forecasts || !forecasts.length) {
        el.innerHTML = '<div class="empty-state">Waiting for first scan...</div>';
        return;
    }
    el.innerHTML = forecasts.map(f => `
        <div class="forecast-card">
            <div class="date">${f.date}</div>
            <div class="temp">${f.high}°F</div>
            <div class="condition">${f.detail}</div>
        </div>
    `).join('');

    const biasBar = document.getElementById('bias-bar');
    if (bias && bias.sample_size > 0) {
        const dir = bias.mean_error > 0 ? 'hot' : 'cold';
        biasBar.innerHTML = `
            <span class="bias-label">NWS Bias:</span>
            <span class="bias-value ${dir}">${bias.mean_error > 0 ? '+' : ''}${bias.mean_error}°F</span>
            <span class="bias-detail">MAE ${bias.abs_error}°F · σ ${bias.std_error}°F · n=${bias.sample_size}</span>
        `;
    } else {
        biasBar.innerHTML = '';
    }
}

function renderHistorical(historical) {
    const section = document.getElementById('historical-section');
    const el = document.getElementById('historical-cards');
    if (!historical || !historical.length) {
        section.style.display = 'none';
        return;
    }
    section.style.display = 'block';
    el.innerHTML = historical.slice(-7).map(h => `
        <div class="forecast-card">
            <div class="date">${h.date}</div>
            <div class="temp" style="background: linear-gradient(135deg, #3b82f6, #6366f1); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;">${h.high}°F</div>
            <div class="condition">Low: ${h.low}°F</div>
        </div>
    `).join('');
}

function renderBuckets(buckets, opportunities) {
    const el = document.getElementById('buckets-grid');
    if (!buckets || !Object.keys(buckets).length) {
        el.innerHTML = '<div class="empty-state">No market data yet</div>';
        return;
    }

    const edgeMap = {};
    if (opportunities) {
        opportunities.forEach(o => {
            edgeMap[o.ticker] = { edge: o.edge, ourProb: o.our_prob };
        });
    }

    let html = '';
    for (const [key, bucketList] of Object.entries(buckets)) {
        html += `<div style="margin-bottom: 8px; color: #64748b; font-size: 0.85em; font-weight: 600;">${key}</div>`;
        for (const b of bucketList) {
            const info = edgeMap[b.ticker];
            const marketPct = (b.yes * 100).toFixed(0);
            const modelPct = info ? info.ourProb.toFixed(0) : '—';
            const edge = info ? info.edge : 0;
            const hasEdge = edge > 3;

            html += `
                <div class="bucket-row ${hasEdge ? 'has-edge' : ''}">
                    <div class="bucket-label">${b.label}</div>
                    <div class="bucket-bar-container">
                        <div class="bucket-bar market" style="width: ${marketPct}%"></div>
                        ${info ? `<div class="bucket-bar model" style="width: ${modelPct}%"></div>` : ''}
                    </div>
                    <div class="bucket-prices">
                        <span class="market-price">Mkt: ${marketPct}%</span>
                        ${info ? ` <span class="model-price">Mdl: ${modelPct}%</span>` : ''}
                    </div>
                    <div class="bucket-edge ${edge > 0 ? 'edge-positive' : 'edge-negative'}">
                        ${edge ? `${edge > 0 ? '+' : ''}${edge.toFixed(1)}%` : ''}
                    </div>
                </div>`;
        }
    }
    el.innerHTML = html;
}

function renderSpreads(spreads) {
    const tbody = document.getElementById('spreads-body');
    if (!spreads || !spreads.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-state">No spreads calculated yet</td></tr>';
        return;
    }
    tbody.innerHTML = spreads.map(s => {
        const evColor = s.ev > 0 ? '#10b981' : '#ef4444';
        const probColor = s.probability >= 60 ? '#10b981' : s.probability >= 40 ? '#f59e0b' : '#ef4444';
        const roiColor = s.roi > 0 ? '#10b981' : '#ef4444';
        return `
        <tr>
            <td><strong>${s.buckets}</strong></td>
            <td>${s.forecast}°F</td>
            <td style="color: ${probColor}; font-weight: 600">${s.probability}%</td>
            <td>$${s.cost.toFixed(2)}</td>
            <td>$${s.profit.toFixed(2)}</td>
            <td style="color: ${evColor}; font-weight: 600">${s.ev > 0 ? '+' : ''}$${s.ev.toFixed(3)}</td>
            <td style="color: ${roiColor}; font-weight: 600">${s.roi > 0 ? '+' : ''}${s.roi.toFixed(1)}%</td>
        </tr>`;
    }).join('');
}

function renderRecommendations(recs) {
    const el = document.getElementById('recommendations');
    if (!recs || !recs.length) {
        el.innerHTML = '<div class="empty-state">Waiting for LLM analysis...</div>';
        return;
    }
    el.innerHTML = recs.map(r => `
        <div class="rec-card ${r.decision}">
            <div class="rec-header">
                <div>
                    <span class="rec-decision ${r.decision}">
                        ${r.decision === 'bet' ? '✅ BET' : '⏭️ SKIP'}
                    </span>
                    <span style="margin-left: 8px; font-weight: 600">${r.spread}</span>
                </div>
                <div class="rec-meta">
                    Confidence: ${r.confidence}% · Prob: ${r.probability}% · Cost: $${r.cost}
                    ${r.decision === 'bet' ? ` · Size: $${r.bet_size}` : ''} · EV: $${r.ev}
                </div>
            </div>
            <div class="rec-reasoning">${r.reasoning}</div>
        </div>
    `).join('');
}

function renderBets(bets) {
    const tbody = document.getElementById('bets-body');
    if (!bets || !bets.length) {
        tbody.innerHTML = '<tr><td colspan="8" class="empty-state">No bets placed</td></tr>';
        return;
    }
    tbody.innerHTML = bets.map(b => `
        <tr>
            <td style="font-size: 0.8em; color: #64748b">${b.time?.split('T')[1]?.split('.')[0] || '—'}</td>
            <td style="font-size: 0.8em">${b.ticker}</td>
            <td><strong>${b.bucket}</strong></td>
            <td style="color: #10b981; font-weight: 600">${b.side.toUpperCase()}</td>
            <td>$${b.price.toFixed(2)}</td>
            <td>${b.qty}</td>
            <td>$${b.cost.toFixed(2)}</td>
            <td style="font-size: 0.8em; color: #94a3b8">${b.reasoning || ''}</td>
        </tr>
    `).join('');
}

function renderScanLog(log) {
    const el = document.getElementById('scan-log');
    if (!log || !log.length) {
        el.textContent = 'Waiting for first scan...';
        return;
    }
    el.textContent = log.join('\n');
    el.scrollTop = el.scrollHeight;
}

async function update() {
    const state = await fetchState();
    if (!state) return;

    const lastScan = document.getElementById('last-scan');
    if (state.last_scan) {
        const d = new Date(state.last_scan.endsWith('Z') ? state.last_scan : state.last_scan + 'Z');
        lastScan.textContent = `Last: ${d.toLocaleTimeString()}`;
    }

    renderForecasts(state.forecasts, state.bias);
    renderHistorical(state.historical);
    renderBuckets(state.buckets, state.opportunities);
    renderSpreads(state.spreads);
    renderRecommendations(state.recommendations);
    renderBets(state.placed_bets);
    renderScanLog(state.scan_log);
}

document.getElementById('rescan-btn').addEventListener('click', async () => {
    const btn = document.getElementById('rescan-btn');
    btn.disabled = true;
    btn.textContent = '⏳ Scanning...';
    await fetch('/api/rescan');
    setTimeout(() => {
        btn.disabled = false;
        btn.textContent = '🔄 Rescan';
    }, 5000);
});

setInterval(update, REFRESH_MS);
update();
