/**
 * admin.js — 管理员仪表盘 (v11 enhanced)
 * 依赖: app.js (currentIdentity, escapeHtml, formatDate, showToast, getRole)
 */
(function() {
    // ============================================================
    // 系统统计
    // ============================================================
    async function loadSystemStats() {
        try {
            var res = await fetch('/api/tickets/stats');
            var data = await res.json();
            if (data.status === 'success' && data.data) {
                var s = data.data;
                var openCount = (s.by_status?.created || 0) + (s.by_status?.assigned || 0) + (s.by_status?.processing || 0);
                var resolvedCount = (s.by_status?.resolved || 0) + (s.by_status?.closed || 0);
                var highCount = (s.by_priority?.P0 || 0) + (s.by_priority?.P1 || 0);
                setStat('admin-stat-total', s.total || 0);
                setStat('admin-stat-open', openCount);
                setStat('admin-stat-resolved', resolvedCount);
                setStat('admin-stat-high', highCount);
            }
        } catch(e) { console.error('Admin stats error:', e); }
    }

    // ============================================================
    // Agent 列表
    // ============================================================
    async function loadAgents() {
        try {
            var res = await fetch('/api/agents/list');
            var agents = await res.json();
            var container = document.getElementById('admin-agent-list');
            if (!container) return;
            if (!agents || agents.length === 0) {
                container.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:13px;">无已注册 Agent</div>';
                return;
            }
            var html = '';
            agents.forEach(function(a) {
                var name = a.name || a.agent_id || 'Unknown';
                var desc = a.description || '';
                html += '<div style="padding:10px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;">' +
                    '<div>' +
                    '<span style="font-weight:500;font-size:13px;">' + escapeHtml(name) + '</span>' +
                    (desc ? '<br><span style="font-size:11px;color:var(--text-secondary);">' + escapeHtml(desc) + '</span>' : '') +
                    '</div>' +
                    '<span style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--success);">' +
                    '<span style="width:6px;height:6px;border-radius:50%;background:var(--success);display:inline-block;"></span>' +
                    'Online</span>' +
                    '</div>';
            });
            container.innerHTML = html;
        } catch(e) { console.error('Admin agents error:', e); }
    }

    // ============================================================
    // 最近工单（全量，不限申请人 — admin 视角）
    // ============================================================
    async function loadRecentTickets() {
        try {
            var res = await fetch('/api/tickets/?role=admin&limit=5&offset=0');
            var data = await res.json();
            var container = document.getElementById('admin-recent-tickets');
            if (!container) return;
            if (data.status === 'success' && data.data && data.data.length > 0) {
                var html = '';
                data.data.forEach(function(t) {
                    var iconMap = { it_fault: 'fa-network-wired', leave: 'fa-umbrella-beach', expense: 'fa-receipt', admin: 'fa-box' };
                    var icon = iconMap[t.ticket_type] || 'fa-file-alt';
                    var statusLabels = { created: '待处理', assigned: '已派发', processing: '处理中', resolved: '已解决', closed: '已关闭' };
                    html += '<div class="approval-card" style="cursor:default;">' +
                        '<div class="approval-icon ' + (t.ticket_type || '') + '"><i class="fas ' + icon + '"></i></div>' +
                        '<div class="approval-body">' +
                        '<div class="ticket-number">' + escapeHtml(t.ticket_number) + '</div>' +
                        '<div class="ticket-title">' + escapeHtml(t.title || '无标题') + '</div>' +
                        '<div class="ticket-meta">' +
                        '<span><i class="fas fa-user"></i> ' + escapeHtml(t.requester_id || '-') + '</span>' +
                        '<span>' + escapeHtml(statusLabels[t.status] || t.status) + '</span>' +
                        '<span>' + escapeHtml(t.priority || '') + '</span>' +
                        '</div></div></div>';
                });
                container.innerHTML = html;
            } else {
                container.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:13px;">暂无工单</div>';
            }
        } catch(e) { console.error('Admin recent tickets error:', e); }
    }

    function setStat(id, value) {
        var el = document.getElementById(id);
        if (el) el.textContent = value;
    }

    // ============================================================
    // 初始化
    // ============================================================
    async function initAdmin() {
        await Promise.all([loadSystemStats(), loadAgents(), loadRecentTickets()]);
    }

    window.initAdmin = initAdmin;
})();
