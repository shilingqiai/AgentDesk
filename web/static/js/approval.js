/**
 * approval.js — 审批门户模块 (v11)
 *
 * 设计原则: visibility ≠ actionability
 * - 链上所有审批人从工单创建起就能看到
 * - 但只有 current_step 对应的人能点按钮
 *
 * 依赖: app.js (currentIdentity, escapeHtml, showToast, renderWorkflowTrail)
 */
(function() {
    var actionableItems = [];    // 可操作的审批项
    var visibleOnlyItems = [];   // 仅可见，等待前置审批
    var historyItems = [];
    var rejectTarget = null;

    function currentApprover() {
        return window.currentIdentity || '王经理';
    }

    function getUser() {
        return currentApprover();
    }

    // ============================================================
    // 数据加载
    // ============================================================

    async function loadPending() {
        try {
            var res = await fetch('/api/approvals/pending?approver=' + encodeURIComponent(currentApprover()));
            var data = await res.json();
            var allItems = (data.success && data.items) ? data.items : [];
            // 按 actionable 拆分为两组
            actionableItems = allItems.filter(function(item) { return item.actionable !== false; });
            visibleOnlyItems = allItems.filter(function(item) { return item.actionable === false; });
        } catch(e) {
            console.error('Failed to load pending:', e);
            actionableItems = [];
            visibleOnlyItems = [];
        }
        renderAll();
        updateStats();

        // Dispatch event so spaApp can update badge count
        try {
            window.dispatchEvent(new CustomEvent('approval-count', {
                detail: { count: actionableItems.length, total: actionableItems.length + visibleOnlyItems.length }
            }));
        } catch(e) {}
    }

    // ============================================================
    // 渲染
    // ============================================================

    function renderAll() {
        renderActionable();
        renderVisibleOnly();
    }

    function renderActionable() {
        var container = document.getElementById('actionable-list');
        var section = document.getElementById('actionable-section');
        if (!container || !section) return;

        if (actionableItems.length === 0) {
            section.style.display = 'none';
            return;
        }
        section.style.display = '';

        container.innerHTML = actionableItems.map(function(item) {
            return buildCard(item, true);
        }).join('');
    }

    function renderVisibleOnly() {
        var container = document.getElementById('visible-only-list');
        var section = document.getElementById('visible-only-section');
        if (!container || !section) return;

        if (visibleOnlyItems.length === 0) {
            section.style.display = 'none';
            return;
        }
        section.style.display = '';

        container.innerHTML = visibleOnlyItems.map(function(item) {
            return buildCard(item, false);
        }).join('');
    }

    function buildCard(item, actionable) {
        var iconClass = item.workflow_type || 'leave';
        var icon = getIcon(iconClass);

        if (actionable) {
            // ── 可操作卡片 ──
            return '<div class="approval-card actionable" id="card-' + item.step_id + '">' +
                '<div class="approval-icon ' + iconClass + '"><i class="fas ' + icon + '"></i></div>' +
                '<div class="approval-body">' +
                '<div class="ticket-number">' + escapeHtml(item.ticket_number) + '</div>' +
                '<div class="ticket-title">' + escapeHtml(item.title || '无标题') + '</div>' +
                '<div class="ticket-meta">' +
                '<span><i class="fas fa-user"></i> ' + escapeHtml(item.requester || '-') + '</span>' +
                getAmountLabel(item) +
                '<span><i class="fas fa-layer-group"></i> 步骤 ' + item.step_order + '/' + item.total_steps + '</span>' +
                '<span class="tag-your-turn">🔔 你的环节</span>' +
                '<span style="cursor:pointer;color:var(--accent);" onclick="window.approvalShowTrail(' + item.ticket_id + ')">' +
                '<i class="fas fa-project-diagram"></i> 查看审批轨迹</span>' +
                '</div></div>' +
                '<div class="approval-actions" style="display:flex;align-items:center;gap:6px;">' +
                '<input type="checkbox" class="approval-checkbox" value="' + item.step_id + '" style="cursor:pointer;" title="选择此项">' +
                '<button class="btn success" onclick="window.approvalApprove(' + item.step_id + ')">' +
                '<i class="fas fa-check"></i> 通过</button>' +
                '<button class="btn danger" onclick="window.openRejectModal(' + item.step_id + ')">' +
                '<i class="fas fa-times"></i> 驳回</button>' +
                '</div></div>';
        } else {
            // ── 仅可见卡片（等待前置审批） ──
            var waitingFor = item.current_approver || '上一级审批人';
            return '<div class="approval-card visible-only" id="card-' + item.step_id + '">' +
                '<div class="approval-icon ' + iconClass + ' muted"><i class="fas ' + icon + '"></i></div>' +
                '<div class="approval-body">' +
                '<div class="ticket-number">' + escapeHtml(item.ticket_number) + '</div>' +
                '<div class="ticket-title">' + escapeHtml(item.title || '无标题') + '</div>' +
                '<div class="ticket-meta">' +
                '<span><i class="fas fa-user"></i> ' + escapeHtml(item.requester || '-') + '</span>' +
                getAmountLabel(item) +
                '<span><i class="fas fa-layer-group"></i> 你的环节: 第 ' + item.step_order + '/' + item.total_steps + ' 步</span>' +
                '<span class="tag-waiting">⏳ 等待 ' + escapeHtml(waitingFor) + ' 审批</span>' +
                '</div></div>' +
                '<div class="approval-actions">' +
                '<button class="btn" disabled title="等待 ' + escapeHtml(waitingFor) + ' 审批">' +
                '<i class="fas fa-check"></i> 通过</button>' +
                '<button class="btn" disabled title="等待 ' + escapeHtml(waitingFor) + ' 审批">' +
                '<i class="fas fa-times"></i> 驳回</button>' +
                '</div></div>';
        }
    }

    function getIcon(type) {
        var map = { leave: 'fa-umbrella-beach', expense: 'fa-receipt', procurement: 'fa-box', purchase: 'fa-box' };
        return map[type] || 'fa-file-alt';
    }

    function getAmountLabel(item) {
        if (item.workflow_type === 'leave' || item.ticket_type === 'leave')
            return '<i class="fas fa-calendar"></i> 请假工单';
        if (item.workflow_type === 'expense' || item.ticket_type === 'expense')
            return '<i class="fas fa-coins"></i> 报销工单';
        if (item.workflow_type === 'procurement' || item.workflow_type === 'purchase')
            return '<i class="fas fa-shopping-cart"></i> 采购工单';
        return '<i class="fas fa-file-alt"></i> ' + escapeHtml(item.workflow_type || item.ticket_type || '工单');
    }

    // ============================================================
    // 操作
    // ============================================================

    async function approveItem(stepId) {
        try {
            var res = await fetch('/api/approvals/approve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    step_id: parseInt(stepId),
                    comment: '',
                    approver_name: getUser(),
                }),
            });
            var data = await res.json();
            if (res.ok && data.success) {
                showToast('审批通过', 'success');
                await loadPending();
            } else {
                showToast(data.detail || '审批失败', 'error');
            }
        } catch(e) {
            showToast('请求失败: ' + e.message, 'error');
        }
    }

    function openRejectModal(stepId) {
        rejectTarget = {stepId: stepId};
        var el = document.getElementById('reject-comment');
        if (el) el.value = '';
        el = document.getElementById('reject-modal');
        if (el) el.classList.add('open');
    }

    function closeRejectModal() {
        rejectTarget = null;
        var el = document.getElementById('reject-modal');
        if (el) el.classList.remove('open');
    }

    async function confirmReject() {
        if (!rejectTarget) return;
        var commentEl = document.getElementById('reject-comment');
        var comment = commentEl ? commentEl.value : '';
        try {
            var res = await fetch('/api/approvals/reject', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    step_id: parseInt(rejectTarget.stepId),
                    comment: comment,
                    approver_name: getUser(),
                }),
            });
            var data = await res.json();
            if (res.ok && data.success) {
                showToast('已驳回', 'error');
                closeRejectModal();
                await loadPending();
            } else {
                showToast(data.detail || '驳回失败', 'error');
            }
        } catch(e) {
            showToast('请求失败: ' + e.message, 'error');
        }
    }

    // ============================================================
    // 审批轨迹
    // ============================================================

    async function showTrail(ticketId) {
        try {
            var res = await fetch('/api/approvals/status/' + ticketId);
            var data = await res.json();
            if (!data.success || !data.workflow) {
                showToast('该工单暂无审批流', 'error');
                return;
            }
            var wf = data.workflow;
            var flow = document.getElementById('trail-flow');
            var section = document.getElementById('trail-section');
            var numberEl = document.getElementById('trail-ticket-number');
            if (flow) {
                flow.innerHTML = '';
                wf.steps.forEach(function(step, i) {
                    var cls = 'pending', icon = step.step_order;
                    if (step.status === 'approved') { cls = 'done'; icon = '✓'; }
                    else if (step.status === 'rejected') { cls = 'rejected'; icon = '✗'; }
                    else if (step.status === 'pending' && wf.status === 'pending') {
                        var prevAllDone = wf.steps.slice(0, i).every(function(s) { return s.status === 'approved'; });
                        if (prevAllDone) { cls = 'current'; icon = '⏳'; }
                    }
                    // Tooltip 数据
                    var tooltip = '';
                    if (step.decided_at) {
                        var d = new Date(step.decided_at);
                        var ts = d.toLocaleString('zh-CN', {month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'});
                        tooltip = ' title="' + escapeHtml(step.approver) + ' — ' + ts + (step.comment ? '\\n' + escapeHtml(step.comment) : '') + '"';
                    }
                    // 耗时
                    var timeDelta = step.decided_at ? '<div class="trail-step-time">' + _fmtDelta(step.created_at, step.decided_at) + '</div>' : '';
                    var roleLabel = _roleLabel(step.approver_role);
                    var arrow = i > 0 ? '<span class="trail-arrow">→</span>' : '';
                    flow.innerHTML += arrow +
                        '<div class="trail-node-wrapper"' + tooltip + '>' +
                        '<div class="trail-node"><div class="node-circle ' + cls + '">' + icon + '</div></div>' +
                        '<div class="trail-node-label">' + escapeHtml(step.approver) + '</div>' +
                        '<div class="trail-node-label" style="font-size:9px;opacity:0.6;">' + escapeHtml(roleLabel) + '</div>' +
                        timeDelta +
                        '</div>';
                });
                var finalCls = 'pending', finalIcon = '\u{1F3C1}';
                if (wf.status === 'approved') { finalCls = 'done'; finalIcon = '✓'; }
                if (wf.status === 'rejected') { finalCls = 'rejected'; finalIcon = '✗'; }
                flow.innerHTML += '<span class="trail-arrow">→</span>' +
                    '<div class="trail-node-wrapper">' +
                    '<div class="trail-node"><div class="node-circle ' + finalCls + '">' + finalIcon + '</div></div>' +
                    '<div class="trail-node-label">完成</div></div>';
            }
            if (section) section.style.display = '';
            if (section) section.scrollIntoView({behavior: 'smooth'});
        } catch(e) {
            showToast('加载审批轨迹失败', 'error');
        }
    }

    /** 计算审批步骤耗时 */
    function _fmtDelta(createdAt, decidedAt) {
        if (!createdAt || !decidedAt) return '';
        try {
            var start = new Date(createdAt);
            var end = new Date(decidedAt);
            var diffMin = Math.round((end - start) / 60000);
            if (diffMin < 60) return diffMin + 'min';
            var h = Math.floor(diffMin / 60);
            var m = diffMin % 60;
            return h + 'h' + (m > 0 ? ' ' + m + 'min' : '');
        } catch(e) { return ''; }
    }

    /** 审批角色 → 中文 */
    function _roleLabel(role) {
        var map = {
            'department_manager': '部门经理',
            'hr': 'HR',
            'finance': '财务',
        };
        return map[role] || role;
    }

    // ============================================================
    // 统计
    // ============================================================

    function updateStats() {
        var el = document.getElementById('stat-pending');
        if (el) el.textContent = actionableItems.length;
        el = document.getElementById('stat-total');
        if (el) el.textContent = actionableItems.length + visibleOnlyItems.length + historyItems.length;

        // 更新空状态
        var empty = document.getElementById('empty-state');
        if (empty) {
            empty.style.display = (actionableItems.length === 0 && visibleOnlyItems.length === 0) ? '' : 'none';
        }
    }

    // ============================================================
    // 初始化
    // ============================================================

    async function initApprovals() {
        await loadPending();
    }

    // ── 批量审批 ──
    async function batchApprove() {
        var checkboxes = document.querySelectorAll('.approval-checkbox:checked');
        if (checkboxes.length === 0) {
            showToast('请至少选择一项审批', 'error');
            return;
        }
        var stepIds = Array.from(checkboxes).map(function(cb) { return parseInt(cb.value); });
        try {
            var res = await fetch('/api/approvals/batch-approve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({step_ids: stepIds, approver_name: getUser()}),
            });
            var data = await res.json();
            if (res.ok && data.success) {
                showToast('批量通过 ' + data.results.success.length + ' 项'
                    + (data.results.failed.length ? '，' + data.results.failed.length + ' 项失败' : ''),
                    'success');
                await loadPending();
            } else {
                showToast(data.detail || '批量审批失败', 'error');
            }
        } catch(e) { showToast('请求失败: ' + e.message, 'error'); }
    }

    function toggleSelectAll(checked) {
        document.querySelectorAll('.approval-checkbox').forEach(function(cb) { cb.checked = checked; });
    }

    // Expose to global
    window.approvalApprove = approveItem;
    window.approvalShowTrail = showTrail;
    window.openRejectModal = openRejectModal;
    window.closeRejectModal = closeRejectModal;
    window.confirmReject = confirmReject;
    window.initApprovals = initApprovals;
    window.batchApprove = batchApprove;
    window.toggleSelectAll = toggleSelectAll;
})();
