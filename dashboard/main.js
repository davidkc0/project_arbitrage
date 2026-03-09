/**
 * Arb Scanner Dashboard — WebSocket client & UI controller
 */

// ── State ──────────────────────────────────────────────────────────────

let ws = null;
let state = {};
let reconnectAttempts = 0;
const MAX_RECONNECT = 10;
const WS_URL = `ws://${location.hostname || '127.0.0.1'}:8000/ws`;

// Selected opportunity for the trade modal
let selectedOpp = null;

// ── WebSocket ──────────────────────────────────────────────────────────

function connect() {
  const statusDot = document.querySelector('.status-dot');
  const statusText = document.querySelector('.status-text');

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    reconnectAttempts = 0;
    statusDot.className = 'status-dot connected';
    statusText.textContent = 'Connected';
    console.log('[WS] Connected');
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'state_update') {
        state = data;
        renderAll();
      } else if (data.type === 'error') {
        console.error('[Server]', data.message);
      }
    } catch (e) {
      console.error('[WS] Parse error:', e);
    }
  };

  ws.onclose = () => {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Disconnected';
    console.log('[WS] Disconnected');

    if (reconnectAttempts < MAX_RECONNECT) {
      reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
      statusText.textContent = `Reconnecting in ${(delay / 1000).toFixed(0)}s...`;
      setTimeout(connect, delay);
    } else {
      statusText.textContent = 'Connection failed';
    }
  };

  ws.onerror = () => {
    statusDot.className = 'status-dot error';
  };
}

function send(action, data = {}) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action, ...data }));
  }
}

// ── Render ──────────────────────────────────────────────────────────────

function renderAll() {
  renderScanner();
  renderOpportunities();
  renderRisk();
  renderVolume();
  renderHistory();
  renderKillSwitch();
}

function renderScanner() {
  const s = state.scanner || {};
  setText('pm-count', s.polymarket_count ?? '—');
  setText('k-count', s.kalshi_count ?? '—');
  setText('pairs-count', s.matched_pairs ?? '—');
}

function renderOpportunities() {
  const opps = state.opportunities || [];
  const tbody = document.getElementById('opp-tbody');
  const badge = document.getElementById('opp-count');
  badge.textContent = opps.length;

  if (opps.length === 0) {
    tbody.innerHTML = `
      <tr class="empty-row">
        <td colspan="8">
          ${state.scanner ? 'No arbitrage opportunities found yet — scanning...' : 'Connecting to scanner...'}
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = opps.map(opp => {
    const edgeClass = opp.net_edge_percent > 0 ? 'edge-positive' : 'edge-negative';
    const confClass = opp.match_confidence >= 80 ? 'confidence-high'
      : opp.match_confidence >= 60 ? 'confidence-medium' : 'confidence-low';

    // Determine which platform is A vs B
    const isAPolymarket = opp.event_a_platform === 'polymarket';
    const pmYes = isAPolymarket ? opp.yes_a : opp.yes_b;
    const kYes = isAPolymarket ? opp.yes_b : opp.yes_a;
    const pmNo = isAPolymarket ? opp.no_a : opp.no_b;
    const kNo = isAPolymarket ? opp.no_b : opp.no_a;

    // Color the cheaper YES price green (that's where you'd buy)
    const yesGreenPM = pmYes < kYes ? 'color: var(--green); font-weight:600' : '';
    const yesGreenK = kYes < pmYes ? 'color: var(--green); font-weight:600' : '';
    // Color the cheaper NO price green too
    const noGreenPM = pmNo < kNo ? 'color: var(--green); font-weight:600' : '';
    const noGreenK = kNo < pmNo ? 'color: var(--green); font-weight:600' : '';

    return `
      <tr data-opp-id="${opp.id}">
        <td>
          <div class="event-question" title="${escapeHtml(opp.event_a_question)}">${escapeHtml(opp.event_a_question)}</div>
        </td>
        <td class="price-cell" style="${yesGreenPM}">$${pmYes.toFixed(3)}</td>
        <td class="price-cell" style="${yesGreenK}">$${kYes.toFixed(3)}</td>
        <td class="price-cell" style="${noGreenPM}">$${pmNo.toFixed(3)}</td>
        <td class="price-cell" style="${noGreenK}">$${kNo.toFixed(3)}</td>
        <td class="edge-cell ${edgeClass}">${opp.net_edge_percent.toFixed(2)}%</td>
        <td>
          <span class="confidence-bar">
            <span class="confidence-bar-fill ${confClass}" style="width: ${opp.match_confidence}%"></span>
          </span>
          <span style="font-size: 11px; color: var(--text-muted)">${opp.match_confidence.toFixed(0)}%</span>
        </td>
        <td>
          <button class="btn-trade" onclick="openTradeModal('${opp.id}')">Trade</button>
        </td>
      </tr>
    `;
  }).join('');
}

function renderRisk() {
  const r = state.risk || {};
  const e = state.execution || {};

  setText('mode-badge', (e.mode || 'semi').toUpperCase());
  document.getElementById('mode-badge').className =
    'mode-badge' + (e.mode === 'auto' ? ' auto-mode' : '');

  const exposurePct = r.max_total_exposure > 0
    ? (r.total_exposure / r.max_total_exposure) * 100 : 0;
  setText('exposure-value', `$${(r.total_exposure || 0).toFixed(0)} / $${(r.max_total_exposure || 500).toFixed(0)}`);
  document.getElementById('exposure-bar').style.width = `${exposurePct}%`;

  const posPct = r.max_positions > 0
    ? (r.open_positions / r.max_positions) * 100 : 0;
  setText('positions-value', `${r.open_positions || 0} / ${r.max_positions || 10}`);
  document.getElementById('positions-bar').style.width = `${posPct}%`;

  setText('max-trade-value', `$${(r.max_position_size || 50).toFixed(0)}`);
}

function renderVolume() {
  // Volume stats come from a separate endpoint; for now show from trade history
  const hist = state.trade_history || [];
  let pmVol = 0, kVol = 0, pmTrades = 0, kTrades = 0;

  hist.forEach(t => {
    if (t.status === 'filled') {
      // Approximate — real stats come from the volume endpoint
      pmVol += (t.total_cost || 0) / 2;
      kVol += (t.total_cost || 0) / 2;
      pmTrades++;
      kTrades++;
    }
  });

  setText('pm-volume', `$${pmVol.toFixed(2)}`);
  setText('pm-trades', `${pmTrades} trades`);
  setText('k-volume', `$${kVol.toFixed(2)}`);
  setText('k-trades', `${kTrades} trades`);
}

function renderHistory() {
  const history = state.trade_history || [];
  const badge = document.getElementById('history-count');
  const list = document.getElementById('history-list');
  badge.textContent = history.length;

  if (history.length === 0) {
    list.innerHTML = '<div class="empty-state">No trades yet</div>';
    return;
  }

  list.innerHTML = history.slice().reverse().map(t => {
    const statusClass = t.status === 'filled' ? 'filled'
      : t.status === 'failed' ? 'failed' : 'pending';
    const profit = t.actual_profit ?? t.expected_profit ?? 0;
    const profitColor = profit >= 0 ? 'var(--green)' : 'var(--red)';

    return `
      <div class="history-item">
        <div>
          <span class="history-status ${statusClass}">${t.status}</span>
          <span style="margin-left: 8px; color: var(--text-muted); font-size: 11px">
            $${(t.total_cost || 0).toFixed(2)}
          </span>
        </div>
        <span class="history-profit" style="color: ${profitColor}">
          ${profit >= 0 ? '+' : ''}$${profit.toFixed(4)}
        </span>
      </div>
    `;
  }).join('');
}

function renderKillSwitch() {
  const btn = document.getElementById('kill-switch-btn');
  const killed = state.risk?.killed || false;

  if (killed) {
    btn.classList.add('engaged');
    btn.textContent = '⚡ RESUME';
  } else {
    btn.classList.remove('engaged');
    btn.textContent = '⚡ KILL';
  }
}

// ── Trade Modal ────────────────────────────────────────────────────────

window.openTradeModal = function (oppId) {
  const opps = state.opportunities || [];
  selectedOpp = opps.find(o => o.id === oppId);
  if (!selectedOpp) return;

  const modal = document.getElementById('trade-modal');
  const body = document.getElementById('modal-body');
  const sizeInput = document.getElementById('trade-size-input');

  const maxSize = state.risk?.max_position_size || 50;
  sizeInput.value = maxSize;

  body.innerHTML = `
    <div class="detail-row">
      <span class="detail-label">Event</span>
      <span class="detail-value" style="max-width: 280px; text-align: right; white-space: normal; font-size: 12px">
        ${escapeHtml(selectedOpp.event_a_question)}
      </span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Buy YES on</span>
      <span class="detail-value">
        <span class="platform-tag ${selectedOpp.buy_yes_platform}">${selectedOpp.buy_yes_platform}</span>
        @ $${selectedOpp.buy_yes_price.toFixed(4)}
      </span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Buy NO on</span>
      <span class="detail-value">
        <span class="platform-tag ${selectedOpp.buy_no_platform}">${selectedOpp.buy_no_platform}</span>
        @ $${selectedOpp.buy_no_price.toFixed(4)}
      </span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Total Cost / Share</span>
      <span class="detail-value">$${selectedOpp.total_cost.toFixed(4)}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Gross Edge</span>
      <span class="detail-value">${(selectedOpp.gross_edge * 100).toFixed(2)}%</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Net Edge (after fees)</span>
      <span class="detail-value edge-positive" style="font-size: 15px">
        ${selectedOpp.net_edge_percent.toFixed(2)}%
      </span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Match Confidence</span>
      <span class="detail-value">${selectedOpp.match_confidence.toFixed(0)}%</span>
    </div>
  `;

  modal.style.display = 'flex';
};

function closeTradeModal() {
  document.getElementById('trade-modal').style.display = 'none';
  selectedOpp = null;
}

function executeTrade() {
  if (!selectedOpp) return;

  const size = parseFloat(document.getElementById('trade-size-input').value);
  if (isNaN(size) || size <= 0) {
    alert('Enter a valid trade size.');
    return;
  }

  send('execute_trade', {
    opp_id: selectedOpp.id,
    size: size,
  });

  closeTradeModal();
}

// ── Kill Switch ────────────────────────────────────────────────────────

document.getElementById('kill-switch-btn').addEventListener('click', () => {
  const isKilled = state.risk?.killed || false;
  if (!isKilled) {
    if (confirm('⚠️ KILL SWITCH: Stop all trading immediately?')) {
      send('kill_switch', { enabled: true });
    }
  } else {
    send('kill_switch', { enabled: false });
  }
});

// ── Modal events ───────────────────────────────────────────────────────

document.getElementById('modal-close').addEventListener('click', closeTradeModal);
document.getElementById('modal-cancel').addEventListener('click', closeTradeModal);
document.getElementById('modal-execute').addEventListener('click', executeTrade);
document.getElementById('trade-modal').addEventListener('click', (e) => {
  if (e.target.id === 'trade-modal') closeTradeModal();
});

// ── Utilities ──────────────────────────────────────────────────────────

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

// ── Init ───────────────────────────────────────────────────────────────

connect();
