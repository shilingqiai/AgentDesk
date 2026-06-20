/**
 * admin.js — 管理员仪表盘 (v13 enhanced)
 *
 * 依赖: app.js (currentIdentity, escapeHtml, formatDate, showToast, getRole)
 *
 * 增强内容:
 * - 动态审批链流程图（替换硬编码卡片）
 * - 实时事件时间线（来自 EventBus DashboardHandler）
 * - 系统统计来自 /api/tickets/dashboard
 * - SLA 超时监控面板
 */

(function() {
    // ============================================================
    // 系统统计（使用新 Dashboard API）
    // ============================================================
    async function loadSystemStats() {
        try {
            var res = await fetch('/api/tickets/dashboard');
            var data = await res.json();
            if (data.status === 'success' && data.data) {
                var s = data.data;
                var openCount = (s.by_status?.created || 0) + (s.by_status?.pending_approval || 0) + (s.by_status?.processing || 0);
                var resolvedCount = (s.by_status?.completed || 0) + (s.by_status?.approved || 0);
                var highCount = (s.by_priority?.P0 || 0) + (s.by_priority?.P1 || 0);
                setStat('admin-stat-total', s.total_tickets || 0);
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
    // 审批链实时状态（动态替换硬编码卡片）
    // ============================================================
    async function loadApprovalChains() {
        try {
            var res = await fetch('/api/approvals/chains');
            var data = await res.json();
            var container = document.getElementById('admin-approval-chains');
            if (!container) return;
            if (!data.success || !data.chains) {
                container.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:13px;">无法加载审批链</div>';
                return;
            }

            var iconMap = { leave: 'fa-umbrella-beach', purchase: 'fa-receipt', it_fault: 'fa-network-wired' };
            var html = '';
            Object.keys(data.chains).forEach(function(key) {
                var chain = data.chains[key];
                var icon = iconMap[key] || 'fa-file-alt';

                html += '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;margin-bottom:10px;">';
                html += '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">';
                html += '<span style="font-weight:600;font-size:14px;"><i class="fas ' + icon + '"></i> ' + escapeHtml(chain.name) + '</span>';
                if (chain.active_count > 0) {
                    html += '<span style="background:#f59e0b;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;">' + chain.active_count + ' 在途</span>';
                } else {
                    html += '<span style="color:var(--text-muted);font-size:11px;">无在途</span>';
                }
                html += '</div>';

                if (chain.steps && chain.steps.length > 0) {
                    html += '<div class="approval-chain-flow">';
                    html += '<div class="chain-node"><div class="chain-node-circle" style="background:#6366f1;border-color:#6366f1;color:#fff;">申</div><div class="chain-node-label">申请人</div></div>';
                    chain.steps.forEach(function(step) {
                        html += '<span class="chain-arrow">→</span>';
                        html += '<div class="chain-node"><div class="chain-node-circle pending">' + (step.name ? step.name[0] : '?') + '</div><div class="chain-node-label">' + escapeHtml(step.name || step.role) + '</div></div>';
                    });
                    html += '<span class="chain-arrow">→</span>';
                    html += '<div class="chain-node"><div class="chain-node-circle pending">✓</div><div class="chain-node-label">完成</div></div>';
                    html += '</div>';
                } else {
                    html += '<div style="color:var(--text-muted);font-size:12px;">无需审批 — 直接创建工单</div>';
                }
                html += '</div>';
            });
            container.innerHTML = html;
        } catch(e) { console.error('Admin chains error:', e); }
    }

    // ============================================================
    // 事件时间线
    // ============================================================
    async function loadEvents() {
        try {
            var res = await fetch('/api/tickets/dashboard');
            var data = await res.json();
            var container = document.getElementById('admin-events-timeline');
            if (!container) return;

            var events = (data.data && data.data.recent_events) || [];
            if (events.length === 0) {
                container.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:13px;">暂无事件</div>';
                return;
            }

            var dotClass = {
                'ticket.created': 'created',
                'ticket.status_changed': 'changed',
                'approved': 'approved',
                'rejected': 'rejected',
            };

            var html = '<div class="events-timeline">';
            events.forEach(function(ev) {
                var dotCls = 'changed';
                if (ev.event && ev.event.indexOf('created') >= 0) dotCls = 'created';
                if (ev.event && ev.event.indexOf('approved') >= 0) dotCls = 'approved';
                if (ev.event && ev.event.indexOf('rejected') >= 0) dotCls = 'rejected';

                html += '<div class="event-item">' +
                    '<div class="event-dot ' + dotCls + '"></div>' +
                    '<span class="event-time">' + escapeHtml(ev.time || '') + '</span>' +
                    '<span class="event-detail">' + escapeHtml(ev.detail || '') + '</span>' +
                    (ev.ticket ? '<span style="color:var(--text-muted);font-size:11px;">' + escapeHtml(ev.ticket) + '</span>' : '') +
                    '</div>';
            });
            html += '</div>';
            container.innerHTML = html;
        } catch(e) { console.error('Admin events error:', e); }
    }

    // ============================================================
    // 最近工单（全量，不限申请人 — admin 视角）
    // ============================================================
    // ============================================================
    // SLA 超时监控
    // ============================================================
    // ============================================================
    // 工具注册中心
    // ============================================================
    // ============================================================
    // ReAct 执行追踪
    // ============================================================
    async function loadTraces() {
        try {
            var res = await fetch('/api/traces/?limit=8');
            var data = await res.json();
            var container = document.getElementById('admin-traces-list');
            if (!container) return;

            if (!data.success || !data.traces || data.traces.length === 0) {
                container.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:13px;">暂无执行追踪 — 发送一条消息后出现</div>';
                return;
            }

            var trackColors = {dynamic:'#6c5ce7', fast:'#00b894', action:'#0984e3', complex:'#fdcb6e', fallback:'#e17055'};
            var html = '';
            data.traces.forEach(function(tr) {
                var trackColor = trackColors[tr.track] || '#636e72';
                var statusIcon = tr.success ? '<i class="fas fa-check-circle" style="color:var(--success);"></i>' :
                    '<i class="fas fa-exclamation-circle" style="color:var(--danger);"></i>';

                html += '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:10px 14px;margin-bottom:8px;">';

                // Header row
                html += '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">' +
                    '<div style="display:flex;align-items:center;gap:8px;">' +
                    '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + trackColor + ';"></span>' +
                    '<span style="font-weight:600;font-size:13px;">' + escapeHtml(tr.user_name || '?') + '</span>' +
                    '<span style="font-size:10px;padding:1px 6px;border-radius:8px;background:' + trackColor + '20;color:' + trackColor + ';">' +
                    escapeHtml(tr.track) + '</span>' +
                    statusIcon +
                    '</div>' +
                    '<span style="font-size:11px;color:var(--text-muted);">' + escapeHtml(tr.timestamp || '') + '</span>' +
                    '</div>';

                // User input
                html += '<div style="font-size:12px;color:var(--text-secondary);margin-bottom:4px;">' +
                    '<i class="fas fa-comment" style="font-size:9px;margin-right:4px;color:var(--text-muted);"></i>' +
                    escapeHtml(tr.user_input || '') + '</div>';

                // Steps — horizontal mini trail
                if (tr.steps && tr.steps.length > 0) {
                    html += '<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;margin:6px 0;">';
                    tr.steps.forEach(function(s, i) {
                        var stepIcon = s.success ? '✅' : '❌';
                        var stepBg = s.success ? '#f0fdf4' : '#fef2f2';
                        html += '<span style="font-size:10px;padding:2px 6px;border-radius:10px;background:' + stepBg +
                            ';border:1px solid ' + (s.success ? '#bbf7d0' : '#fecaca') + ';">' +
                            stepIcon + ' ' + escapeHtml(s.tool || '?') + '</span>';
                        if (i < tr.steps.length - 1) {
                            html += '<span style="color:var(--text-muted);font-size:8px;">→</span>';
                        }
                    });
                    html += '</div>';
                }

                // Footer stats
                html += '<div style="font-size:10px;color:var(--text-muted);display:flex;gap:12px;">' +
                    '<span><i class="fas fa-sync-alt"></i> ' + (tr.iterations || 0) + ' 轮</span>' +
                    '<span><i class="fas fa-wrench"></i> ' + (tr.tool_count || 0) + ' 工具</span>' +
                    (tr.error ? '<span style="color:var(--danger);">' + escapeHtml(tr.error) + '</span>' : '') +
                    '</div>';

                html += '</div>';
            });

            // Total stored count
            if (data.total_stored > 0) {
                html += '<div style="text-align:center;font-size:11px;color:var(--text-muted);padding:4px;">' +
                    '共 ' + data.total_stored + ' 条追踪（重启清空）</div>';
            }

            container.innerHTML = html;
        } catch(e) { console.error('Traces load error:', e); }
    }

    async function loadTools() {
        try {
            var res = await fetch('/api/tools/');
            var data = await res.json();
            var container = document.getElementById('admin-tools-list');
            if (!container) return;

            if (!data.success || !data.tools) {
                container.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:13px;">无法加载工具列表</div>';
                return;
            }

            var catLabels = {ops:'运营操作', knowledge:'知识检索', ticket:'工单相关', external:'外部服务', internal:'内部工具'};
            var catColors = {ops:'#6366f1', knowledge:'#10b981', ticket:'#f59e0b', external:'#ef4444', internal:'#6b7280'};

            // 按分类分组
            var groups = {};
            data.tools.forEach(function(t) {
                var cat = t.category || 'general';
                if (!groups[cat]) groups[cat] = [];
                groups[cat].push(t);
            });

            var html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;">';
            Object.keys(groups).sort().forEach(function(cat) {
                var tools = groups[cat];
                html += '<div style="flex:1;min-width:200px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;">';
                html += '<div style="padding:6px 12px;background:' + (catColors[cat] || '#6b7280') + ';color:#fff;font-size:11px;font-weight:600;">' +
                    (catLabels[cat] || cat) + ' <span style="opacity:0.7;">(' + tools.length + ')</span></div>';
                tools.forEach(function(t) {
                    html += '<div style="padding:8px 12px;border-bottom:1px solid var(--border);">' +
                        '<div style="font-size:13px;font-weight:500;margin-bottom:2px;">' +
                        '<i class="fas fa-wrench" style="font-size:10px;margin-right:4px;color:var(--text-muted);"></i>' +
                        escapeHtml(t.name) + '</div>' +
                        '<div style="font-size:11px;color:var(--text-secondary);line-height:1.3;">' + escapeHtml(t.description) + '</div>' +
                        '<div style="margin-top:4px;display:flex;gap:4px;flex-wrap:wrap;">';
                    Object.keys(t.parameters || {}).forEach(function(p) {
                        var param = t.parameters[p];
                        html += '<span style="font-size:9px;padding:1px 5px;background:var(--bg-input);border-radius:3px;color:var(--text-muted);">' +
                            escapeHtml(p) + ': ' + escapeHtml(param.type || 'str') +
                            (param.required ? ' *' : '') + '</span>';
                    });
                    html += '</div></div>';
                });
                html += '</div>';
            });
            html += '</div>';
            container.innerHTML = html;
        } catch(e) { console.error('Tools load error:', e); }
    }

    async function loadSlaStatus() {
        try {
            var res = await fetch('/api/tickets/admin/sla/status');
            var data = await res.json();
            var container = document.getElementById('admin-sla-status');
            if (!container) return;

            if (data.status !== 'success' || !data.data) {
                container.innerHTML = '<div style="padding:12px;color:var(--text-secondary);font-size:13px;">无法加载 SLA 状态</div>';
                return;
            }

            var s = data.data;
            var breachedCount = s.breached_count || 0;
            var warningCount = s.warning_count || 0;
            var activeCount = s.active_sla_count || 0;
            var breachedTickets = s.breached_tickets || [];
            var approvalDeadlines = s.approval_deadlines || [];

            // SLA 概览行
            var html = '<div style="display:flex;gap:16px;margin-bottom:12px;">' +
                '<div class="stat-card" style="flex:1;">' +
                '<div class="num rejected" id="sla-breached-count">' + breachedCount + '</div>' +
                '<div class="label">超时工单</div></div>' +
                '<div class="stat-card" style="flex:1;">' +
                '<div class="num pending" id="sla-warning-count">' + warningCount + '</div>' +
                '<div class="label">预警项</div></div>' +
                '<div class="stat-card" style="flex:1;">' +
                '<div class="num total" id="sla-active-count">' + activeCount + '</div>' +
                '<div class="label">监控中</div></div></div>';

            // 超时工单列表
            if (breachedTickets.length > 0) {
                html += '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;">';
                html += '<div style="padding:8px 16px;background:#fff3f3;border-bottom:1px solid #fecaca;font-size:12px;font-weight:600;color:#dc2626;">' +
                    '<i class="fas fa-exclamation-triangle"></i> 超时工单 (' + breachedTickets.length + ')</div>';
                breachedTickets.forEach(function(t) {
                    html += '<div style="padding:8px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;">' +
                        '<span style="font-size:13px;"><strong>' + escapeHtml(t.ticket_number) + '</strong> ' +
                        escapeHtml(t.title || '') + '</span>' +
                        '<span style="font-size:11px;color:var(--danger);">' +
                        escapeHtml(t.rule_label || '') + ' ' + (t.elapsed_h || 0).toFixed(1) + 'h/' + (t.duration_h || 0) + 'h</span>' +
                        '</div>';
                });
                html += '</div>';
            } else {
                html += '<div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:12px 16px;color:var(--success);font-size:13px;">' +
                    '<i class="fas fa-check-circle"></i> 所有在途工单均未超时</div>';
            }

            // 审批节点 SLA
            if (approvalDeadlines.length > 0) {
                html += '<div style="margin-top:12px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;">';
                html += '<div style="padding:8px 16px;background:#fffbeb;border-bottom:1px solid #fde68a;font-size:12px;font-weight:600;color:#d97706;">' +
                    '<i class="fas fa-clock"></i> 审批节点 SLA (' + approvalDeadlines.length + ')</div>';
                approvalDeadlines.forEach(function(a) {
                    var breachedCls = a.is_breached ? 'color:#dc2626;' : 'color:var(--text-secondary);';
                    var icon = a.is_breached ? '<i class="fas fa-exclamation-circle" style="color:#dc2626;"></i> ' : '';
                    html += '<div style="padding:8px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;">' +
                        '<span style="font-size:13px;">' + icon + escapeHtml(a.ticket_number) +
                        ' — ' + escapeHtml(a.approver) + ' (步骤' + a.step_order + '/' + a.total_steps + ')</span>' +
                        '<span style="font-size:11px;' + breachedCls + '">' +
                        (a.elapsed_h || 0).toFixed(1) + 'h/' + (a.duration_h || 0) + 'h</span>' +
                        '</div>';
                });
                html += '</div>';
            }

            container.innerHTML = html;
        } catch(e) { console.error('SLA status error:', e); }
    }

    async function loadRecentTickets() {
        try {
            var res = await fetch('/api/tickets/?role=admin&limit=5&offset=0');
            var data = await res.json();
            var container = document.getElementById('admin-recent-tickets');
            if (!container) return;
            if (data.status === 'success' && data.data && data.data.length > 0) {
                var statusLabels = {
                    created: '已创建', pending_approval: '待审批', approved: '已通过',
                    rejected: '已驳回', processing: '执行中', completed: '已完成'
                };
                var html = '';
                data.data.forEach(function(t) {
                    var iconMap = { it_fault: 'fa-network-wired', leave: 'fa-umbrella-beach', expense: 'fa-receipt', admin: 'fa-box' };
                    var icon = iconMap[t.ticket_type] || 'fa-file-alt';
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
        await Promise.all([
            loadSystemStats(),
            loadAgents(),
            loadTools(),
            loadTraces(),
            loadSlaStatus(),
            loadApprovalChains(),
            loadEvents(),
            loadRecentTickets(),
        ]);
    }

    window.initAdmin = initAdmin;
})();
