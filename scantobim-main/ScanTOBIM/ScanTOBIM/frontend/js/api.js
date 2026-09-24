/**
 * api.js — ScanToBIM shared API client + auth utilities
 *
 * Usage: include this script before Alpine.js on every page.
 * All API calls go through the `api` object.
 * Auth state is stored in localStorage under `stb_token` + `stb_user`.
 */

// ── Config ────────────────────────────────────────────────────────────────────

const API_BASE = (() => {
  // In production (same-origin via Caddy), use empty string (relative URLs).
  // In local dev with `uvicorn agent.main:app --port 8765`, point to that port.
  const { hostname, port, protocol } = window.location;
  if (hostname === 'localhost' && port !== '8765') {
    return `${protocol}//localhost:8765`;
  }
  return ''; // same origin
})();

// ── Auth helpers ──────────────────────────────────────────────────────────────

const auth = {
  token()   { return localStorage.getItem('stb_token'); },
  user()    { try { return JSON.parse(localStorage.getItem('stb_user') || 'null'); } catch { return null; } },
  isLoggedIn() { return !!this.token(); },

  save(token, user) {
    localStorage.setItem('stb_token', token);
    localStorage.setItem('stb_user', JSON.stringify(user));
  },

  logout() {
    localStorage.removeItem('stb_token');
    localStorage.removeItem('stb_user');
    window.location.href = '/ui/login.html';
  },

  /** Call on every protected page — redirects to login if no token. */
  require() {
    if (!this.isLoggedIn()) {
      window.location.href = '/ui/login.html';
      return false;
    }
    return true;
  },

  /** Role checks — matches server-side roles */
  isEngineer()   { const u = this.user(); return u && ['engineer', 'admin'].includes(u.role); },
  isBimManager() { const u = this.user(); return u && ['bim_manager', 'admin'].includes(u.role); },
  isAdmin()      { const u = this.user(); return u && u.role === 'admin'; },
};

// ── HTTP helpers ──────────────────────────────────────────────────────────────

function _headers(extra = {}) {
  const h = { 'Content-Type': 'application/json', ...extra };
  const t = auth.token();
  if (t) h['Authorization'] = `Bearer ${t}`;
  return h;
}

async function _handleResp(resp) {
  if (resp.status === 401) { auth.logout(); return null; }
  if (resp.status === 204) return null;
  const data = await resp.json().catch(() => ({ detail: resp.statusText }));
  if (!resp.ok) throw { status: resp.status, detail: data.detail || JSON.stringify(data) };
  return data;
}

// ── Core API ──────────────────────────────────────────────────────────────────

const api = {

  // ── Auth ───────────────────────────────────────────────────────────────────

  async login(username, password) {
    const body = new URLSearchParams({ username, password });
    const resp = await fetch(`${API_BASE}/auth/token`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    });
    if (!resp.ok) throw { status: resp.status, detail: 'Invalid credentials' };
    const data = await resp.json();
    // Also fetch the user profile
    const me = await fetch(`${API_BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${data.access_token}` },
    }).then(r => r.json());
    auth.save(data.access_token, me);
    return { token: data.access_token, user: me };
  },

  // ── Health ─────────────────────────────────────────────────────────────────

  async health()       { return _handleResp(await fetch(`${API_BASE}/health`)); },
  async bridgeHealth() {
    try {
      return await _handleResp(await fetch(`${API_BASE}/bridge/health`));
    } catch { return { connected: false }; }
  },

  // ── Sessions ───────────────────────────────────────────────────────────────

  async listSessions() {
    return _handleResp(await fetch(`${API_BASE}/sessions`, { headers: _headers() }));
  },

  async createSession(siteName, zoneId = 'zone-001', useSynthetic = true) {
    return _handleResp(await fetch(`${API_BASE}/sessions`, {
      method:  'POST',
      headers: _headers(),
      body:    JSON.stringify({ site_name: siteName, zone_id: zoneId, use_synthetic: useSynthetic }),
    }));
  },

  async getSession(id) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${id}`, { headers: _headers() }));
  },

  async getCde(id) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${id}/cde`, { headers: _headers() }));
  },

  async transitionCde(id, targetState, notes = '') {
    return _handleResp(await fetch(`${API_BASE}/sessions/${id}/cde/transition`, {
      method:  'POST',
      headers: _headers(),
      body:    JSON.stringify({ target_state: targetState, notes }),
    }));
  },

  async exportIfc(id) {
    const resp = await fetch(`${API_BASE}/sessions/${id}/export/ifc`, { headers: _headers() });
    if (!resp.ok) throw { status: resp.status, detail: 'IFC export failed' };
    return resp; // caller handles blob download
  },

  async getRevitResults(id) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${id}/revit-results`, { headers: _headers() }));
  },

  async getDrp(id) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${id}/drp`, { headers: _headers() }));
  },

  // ── Safety Gates ───────────────────────────────────────────────────────────

  async listGates(status = 'PENDING') {
    const qs = status ? `?status=${status}` : '';
    return _handleResp(await fetch(`${API_BASE}/safety-gates${qs}`, { headers: _headers() }));
  },

  async getGate(gateId) {
    return _handleResp(await fetch(`${API_BASE}/safety-gates/${gateId}`, { headers: _headers() }));
  },

  async decideGate(gateId, approved, approverUpn, comments = '') {
    return _handleResp(await fetch(`${API_BASE}/safety-gates/${gateId}/decision`, {
      method:  'POST',
      headers: _headers(),
      body:    JSON.stringify({ approved, approver_upn: approverUpn, comments }),
    }));
  },

  async notifyTeams(sessionId) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${sessionId}/notify-teams`, {
      method:  'POST',
      headers: _headers(),
    }));
  },

  // ── NCRs ───────────────────────────────────────────────────────────────────

  async listNcrs(sessionId) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${sessionId}/ncrs`, { headers: _headers() }));
  },

  async raiseNcr(sessionId, elementId, description, severity = 'MAJOR') {
    return _handleResp(await fetch(`${API_BASE}/sessions/${sessionId}/ncrs`, {
      method:  'POST',
      headers: _headers(),
      body:    JSON.stringify({ element_id: elementId, description, severity, raised_by: auth.user()?.username || 'user' }),
    }));
  },

  async resolveNcr(ncrId, resolvedBy, resolution) {
    return _handleResp(await fetch(`${API_BASE}/ncrs/${ncrId}/resolve`, {
      method:  'PATCH',
      headers: _headers(),
      body:    JSON.stringify({ resolved_by: resolvedBy, resolution }),
    }));
  },

  // ── Audit ──────────────────────────────────────────────────────────────────

  async getAudit(sessionId) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${sessionId}/audit`, { headers: _headers() }));
  },

  async verifyChain(sessionId) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${sessionId}/chain`, { headers: _headers() }));
  },


  // ── Plugin Ingest (Sprint P4) ───────────────────────────────────────────────

  async ingestElements(sessionId, segments, scanMetadata = {}, source = 'revit_plugin') {
    return _handleResp(await fetch(`${API_BASE}/sessions/${sessionId}/ingest-elements`, {
      method:  'POST',
      headers: _headers(),
      body:    JSON.stringify({ segments, scan_metadata: scanMetadata, source }),
    }));
  },

  // ── Quality deliverables (P2.2) ─────────────────────────────────────────────

  /** USIBD LOA v3.1 session report */
  async getLoaReport(sessionId) {
    return _handleResp(await fetch(`${API_BASE}/sessions/${sessionId}/loa-report`,
      { headers: _headers() }));
  },

  /** Deviation heatmap — returns Response (caller handles svg/json) */
  async getDeviationHeatmap(sessionId, { view = 'plan', colourBy = 'deviation',
                                         format = 'svg' } = {}) {
    const qs = new URLSearchParams({ view, colour_by: colourBy, format });
    const resp = await fetch(
      `${API_BASE}/sessions/${sessionId}/deviation-heatmap?${qs}`,
      { headers: _headers() },
    );
    if (!resp.ok) throw { status: resp.status, detail: 'heatmap failed' };
    return resp;
  },

  /** Registration QA report (RICS / PAS 128) */
  async getRegistrationReport(sessionId, format = 'json') {
    const resp = await fetch(
      `${API_BASE}/sessions/${sessionId}/registration-report?format=${format}`,
      { headers: _headers() },
    );
    if (!resp.ok) throw { status: resp.status, detail: 'registration report failed' };
    if (format === 'html') return resp;
    return resp.json();
  },

  /** Deviation report (JSON) */
  async getDeviationReport(sessionId) {
    return _handleResp(await fetch(
      `${API_BASE}/sessions/${sessionId}/deviation-report`,
      { headers: _headers() },
    ));
  },

  /** Full handover bundle (.zip stream) */
  async getHandoverBundle(sessionId) {
    const resp = await fetch(
      `${API_BASE}/sessions/${sessionId}/handover-bundle`,
      { headers: _headers() },
    );
    if (!resp.ok) throw { status: resp.status, detail: 'handover bundle failed' };
    return resp;
  },

  // ── NQA-1 three-party signatures (P2.4++) ───────────────────────────────────

  async getNqa1Signatures(sessionId) {
    return _handleResp(await fetch(
      `${API_BASE}/sessions/${sessionId}/nqa1-signatures`,
      { headers: _headers() },
    ));
  },

  async signNqa1(sessionId, role, signerUpn, signerName, comments = '') {
    return _handleResp(await fetch(
      `${API_BASE}/sessions/${sessionId}/nqa1-signatures`,
      {
        method:  'POST',
        headers: _headers(),
        body:    JSON.stringify({
          role, signer_upn: signerUpn, signer_name: signerName, comments,
        }),
      },
    ));
  },

  async revokeNqa1Signature(signatureId, reason) {
    return _handleResp(await fetch(
      `${API_BASE}/nqa1-signatures/${signatureId}/revoke`,
      {
        method:  'PATCH',
        headers: _headers(),
        body:    JSON.stringify({ reason }),
      },
    ));
  },

  // ── Upload ─────────────────────────────────────────────────────────────────

  async uploadScan(sessionId, file) {
    const form = new FormData();
    form.append('file', file);
    const resp = await fetch(`${API_BASE}/sessions/${sessionId}/upload`, {
      method:  'POST',
      headers: { Authorization: `Bearer ${auth.token()}` }, // no Content-Type — let browser set boundary
      body:    form,
    });
    return _handleResp(resp);
  },

  async getJob(jobId) {
    return _handleResp(await fetch(`${API_BASE}/jobs/${jobId}`, { headers: _headers() }));
  },

  // ── SSE ────────────────────────────────────────────────────────────────────

  /** Returns an EventSource for session progress events. */
  progressStream(sessionId) {
    return new EventSource(`${API_BASE}/sessions/${sessionId}/progress`);
  },
};

// ── UI helpers ────────────────────────────────────────────────────────────────

/** Safety category badge HTML */
function safetyCategoryBadge(cat) {
  const map = {
    SC1: 'bg-red-100 text-red-800 border border-red-300',
    SC2: 'bg-amber-100 text-amber-800 border border-amber-300',
    SC3: 'bg-blue-100 text-blue-800 border border-blue-300',
    NS:  'bg-green-100 text-green-800 border border-green-300',
  };
  const cls = map[cat] || 'bg-gray-100 text-gray-700';
  return `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}">${cat}</span>`;
}

/** CDE state badge */
function cdeStateBadge(state) {
  const map = {
    WIP:       'bg-yellow-100 text-yellow-800',
    SHARED:    'bg-blue-100 text-blue-800',
    PUBLISHED: 'bg-green-100 text-green-800',
    FAILED:    'bg-red-100 text-red-800',
  };
  const cls = map[state] || 'bg-gray-100 text-gray-700';
  return `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}">${state}</span>`;
}

/** NCR severity badge */
function ncrSeverityBadge(sev) {
  const map = {
    CRITICAL: 'bg-red-100 text-red-800',
    MAJOR:    'bg-orange-100 text-orange-800',
    MINOR:    'bg-yellow-100 text-yellow-800',
  };
  const cls = map[sev] || 'bg-gray-100 text-gray-700';
  return `<span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}">${sev || '-'}</span>`;
}

/** Format ISO timestamp to local readable */
function fmtDate(iso) {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

/** Short ID (first 8 chars) */
function shortId(id) {
  return id ? id.slice(0, 8) : '—';
}

/** Download a blob as a file */
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = document.createElement('a');
  a.href    = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Show a temporary toast notification */
function toast(message, type = 'info') {
  const colors = { info: 'bg-blue-600', success: 'bg-green-600', error: 'bg-red-600', warn: 'bg-amber-500' };
  const div    = document.createElement('div');
  div.className = `fixed bottom-4 right-4 z-50 px-4 py-3 rounded shadow-lg text-white text-sm ${colors[type] || colors.info}`;
  div.textContent = message;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), 3500);
}
