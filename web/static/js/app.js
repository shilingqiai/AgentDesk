/**
 * app.js — 企业智能服务台 共享核心
 *
 * 身份管理（4 固定身份）、Tab 可见性、全局工具函数、
 * 审批轨迹渲染、sessionStorage 消息持久化。
 */

// ============================================================
// 4 个固定身份
// ============================================================
const IDENTITIES = [
    { key: '张三',  label: '👤 张三 (员工)',   role: 'employee', tabs: ['chat', 'tickets'] },
    { key: '王经理', label: '👔 王经理 (经理)', role: 'manager',  tabs: ['approval', 'tickets'] },
    { key: '李HR',   label: '💼 李HR (HR)',    role: 'hr',       tabs: ['approval', 'tickets'] },
    { key: 'Admin',  label: '🔸 Admin',        role: 'admin',    tabs: ['chat', 'tickets', 'approval', 'admin'] },
];

const TABS = [
    { key: 'chat',     label: '💬 Chat',       icon: 'fa-comments' },
    { key: 'tickets',  label: '📋 Tickets',    icon: 'fa-clipboard-list' },
    { key: 'approval', label: '👔 Approval',   icon: 'fa-user-check' },
    { key: 'admin',    label: '⚙ Admin',       icon: 'fa-cog' },
];

// 审批人 → 角色映射
const APPROVER_ROLES = {
    '王经理': 'department_manager',
    '李HR': 'hr',
    '赵财务': 'finance',
};

// ============================================================
// localStorage / sessionStorage 键名
// ============================================================
const IDENTITY_KEY   = 'service_desk_identity';
const THREAD_KEY     = 'service_desk_thread_id';
const MSG_STORE_KEY  = 'service_desk_messages';

// ============================================================
// 身份管理
// ============================================================
function loadIdentity() {
    try {
        const saved = localStorage.getItem(IDENTITY_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            // 验证是否在已知身份中
            const found = IDENTITIES.find(i => i.key === parsed.user_name);
            if (found) return parsed.user_name;
        }
    } catch (e) {}
    return '张三'; // 默认
}

function saveIdentity(identityKey) {
    try {
        localStorage.setItem(IDENTITY_KEY, JSON.stringify({
            user_name: identityKey,
            role: getRole(identityKey),
        }));
    } catch (e) {}
}

function getRole(identityKey) {
    const id = IDENTITIES.find(i => i.key === identityKey);
    return id ? id.role : 'employee';
}

function getVisibleTabs(identityKey) {
    const id = IDENTITIES.find(i => i.key === identityKey);
    if (!id) return TABS.filter(t => t.key === 'chat');
    return TABS.filter(t => id.tabs.includes(t.key));
}

// ============================================================
// Tab 状态
// ============================================================
function loadTab(identityKey) {
    try {
        const saved = sessionStorage.getItem('service_desk_tab');
        if (saved) {
            const visible = getVisibleTabs(identityKey);
            if (visible.find(t => t.key === saved)) return saved;
        }
    } catch (e) {}
    const visible = getVisibleTabs(identityKey);
    return visible.length > 0 ? visible[0].key : 'chat';
}

function saveTab(tabKey) {
    try { sessionStorage.setItem('service_desk_tab', tabKey); } catch (e) {}
}

// ============================================================
// threadId 持久化
// ============================================================
function getThreadId() {
    let tid = localStorage.getItem(THREAD_KEY);
    if (!tid) {
        tid = 'web_' + Date.now();
        localStorage.setItem(THREAD_KEY, tid);
    }
    return tid;
}

function resetThreadId() {
    const tid = 'web_' + Date.now();
    localStorage.setItem(THREAD_KEY, tid);
    return tid;
}

// ============================================================
// sessionStorage 消息持久化
// ============================================================
function getStoredMessages() {
    try {
        const raw = sessionStorage.getItem(MSG_STORE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch (e) { return []; }
}

function saveStoredMessages(msgs) {
    try {
        const trimmed = msgs.slice(-50);
        sessionStorage.setItem(MSG_STORE_KEY, JSON.stringify(trimmed));
    } catch (e) {
        try {
            sessionStorage.removeItem(MSG_STORE_KEY);
            sessionStorage.setItem(MSG_STORE_KEY, JSON.stringify(msgs.slice(-20)));
        } catch (e2) {}
    }
}

function appendToMsgLog(msg) {
    const msgs = getStoredMessages();
    msgs.push(msg);
    saveStoredMessages(msgs);
}

function clearStoredMessages() {
    try { sessionStorage.removeItem(MSG_STORE_KEY); } catch (e) {}
}

// ============================================================
// 工具函数
// ============================================================
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}

function showToast(message, type, duration) {
    type = type || 'success';
    duration = duration || 3000;
    let toast = document.getElementById('global-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'global-toast';
        toast.style.cssText = 'position:fixed;top:20px;right:20px;z-index:9999;padding:12px 20px;border-radius:8px;color:#fff;font-size:14px;box-shadow:0 8px 24px rgba(0,0,0,0.12);animation:slideIn 0.3s ease;';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.background = type === 'success' ? '#00b894' : type === 'error' ? '#e17055' : '#6c5ce7';
    toast.style.display = '';
    clearTimeout(toast._timeout);
    toast._timeout = setTimeout(() => { toast.style.display = 'none'; }, duration);
}

function formatDate(d) {
    if (!d) return '';
    const date = typeof d === 'string' ? new Date(d) : d;
    return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
}

function formatDateFull(d) {
    if (!d) return '';
    const date = typeof d === 'string' ? new Date(d) : d;
    return date.toLocaleDateString('zh-CN', { year: 'numeric', month: 'short', day: 'numeric' });
}

// ============================================================
// 审批轨迹渲染（ticket tab + approval tab 共享）
// ============================================================
function renderWorkflowTrail(wf) {
    if (!wf || !wf.steps) return '<div class="text-muted">无审批流</div>';

    let html = '<div class="wf-trail">';
    wf.steps.forEach((step, i) => {
        let cls = 'pending';
        let icon = step.step_order;
        if (step.status === 'approved')  { cls = 'done';     icon = '✓'; }
        if (step.status === 'rejected')  { cls = 'rejected'; icon = '✗'; }
        // 找到当前待审批步骤
        if (step.status === 'pending' && wf.status === 'pending') {
            const prevAllDone = wf.steps.slice(0, i).every(s => s.status === 'approved');
            if (prevAllDone) { cls = 'current'; icon = '⏳'; }
        }

        const arrow = i > 0 ? '<span class="wf-arrow">→</span>' : '';
        html += `${arrow}
        <div class="wf-node">
            <div class="wf-dot ${cls}">${icon}</div>
            <div class="wf-node-label">${escapeHtml(step.approver)}</div>
        </div>`;
    });

    // 最终节点
    let finalCls = 'pending';
    let finalIcon = '🏁';
    if (wf.status === 'approved')  { finalCls = 'done';     finalIcon = '✓'; }
    if (wf.status === 'rejected')  { finalCls = 'rejected'; finalIcon = '✗'; }

    html += `
        <span class="wf-arrow">→</span>
        <div class="wf-node">
            <div class="wf-dot ${finalCls}">${finalIcon}</div>
            <div class="wf-node-label">完成</div>
        </div>`;
    html += '</div>';

    // 驳回备注
    const rejectedStep = wf.steps.find(s => s.status === 'rejected');
    if (rejectedStep && rejectedStep.comment) {
        html += `<div class="text-danger" style="margin-top:8px;font-size:13px;">
            ❌ 驳回理由：${escapeHtml(rejectedStep.comment)}
        </div>`;
    }
    return html;
}
