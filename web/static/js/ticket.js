/**
 * ticket.js — 工单管理模块
 * 依赖: app.js (currentIdentity, getRole, escapeHtml, showToast, renderWorkflowTrail)
 */
document.addEventListener('alpine:init', () => {
    Alpine.data('ticketsApp', () => {
        // Use window.currentIdentity from app.js
        let _self;

        return {
            tickets: [],
            stats: { total: 0, open: 0, resolved: 0, high_priority: 0 },
            totalCount: 0,
            currentPage: 1,
            totalPages: 1,
            pageSize: 10,
            expandedId: null,
            loading: false,
            filters: { type: '', status: '', priority: '' },
            toast: { show: false, message: '', type: 'success' },

            get currentUser() { return window.currentIdentity || '张三'; },
            get currentRole() { return window.getRole ? window.getRole(window.currentIdentity) : 'employee'; },
            get isAdmin() { return this.currentRole === 'admin'; },

            async init() {
                _self = this;
                await this.loadStats();
                await this.loadTickets();
            },

            async loadStats() {
                try {
                    const res = await fetch('/api/tickets/stats');
                    const data = await res.json();
                    if (data.status === 'success' && data.data) {
                        const d = data.data;
                        this.stats = {
                            total: d.total || 0,
                            open: (d.by_status?.created || 0) + (d.by_status?.assigned || 0) + (d.by_status?.processing || 0),
                            resolved: (d.by_status?.resolved || 0) + (d.by_status?.closed || 0),
                            high_priority: (d.by_priority?.P0 || 0) + (d.by_priority?.P1 || 0),
                        };
                    }
                } catch (e) { console.error('Stats load error:', e); }
            },

            async loadTickets() {
                this.loading = true;
                try {
                    const params = new URLSearchParams();
                    if (this.filters.type) params.set('ticket_type', this.filters.type);
                    if (this.filters.status) params.set('status', this.filters.status);
                    if (this.filters.priority) params.set('priority', this.filters.priority);
                    params.set('limit', String(this.pageSize));
                    params.set('offset', String((this.currentPage - 1) * this.pageSize));
                    // 身份注入：员工只看自己的工单
                    params.set('user_name', this.currentUser);
                    params.set('role', this.currentRole);
                    if (!this.isAdmin) {
                        params.set('requester_id', this.currentUser);
                    }
                    const res = await fetch('/api/tickets/?' + params);
                    const data = await res.json();
                    if (data.status === 'success') {
                        this.tickets = data.data || [];
                        this.totalCount = data.total || 0;
                        this.totalPages = Math.ceil(this.totalCount / this.pageSize) || 1;
                    }
                } catch (e) { console.error('Tickets load error:', e); }
                this.loading = false;
            },

            applyFilters() {
                this.currentPage = 1;
                this.loadTickets();
            },

            async expandTicket(ticket) {
                if (this.expandedId === ticket.id) {
                    this.expandedId = null;
                    return;
                }
                this.expandedId = ticket.id;
                if (!ticket._workflowLoaded) {
                    ticket._workflow = false;
                    ticket._workflowHtml = '';
                    try {
                        const res = await fetch('/api/approvals/status/' + ticket.id);
                        const data = await res.json();
                        if (data.success && data.workflow) {
                            ticket._workflow = true;
                            ticket._workflowHtml = window.renderWorkflowTrail(data.workflow);
                        }
                    } catch (e) {}
                    ticket._workflowLoaded = true;
                }
            },

            async updateStatus(ticketId, newStatus) {
                try {
                    const params = new URLSearchParams();
                    params.set('user_name', this.currentUser);
                    params.set('role', this.currentRole);
                    const res = await fetch('/api/tickets/' + ticketId + '/status?' + params, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ status: newStatus }),
                    });
                    if (res.ok) {
                        this.showToast('✅ 状态已更新', 'success');
                        await this.loadTickets();
                    } else {
                        const err = await res.json();
                        this.showToast(err.detail || '更新失败', 'error');
                    }
                } catch (e) { this.showToast('❌ 更新失败', 'error'); }
            },

            goPage(p) {
                if (p >= 1 && p <= this.totalPages) {
                    this.currentPage = p;
                    this.loadTickets();
                }
            },

            get visiblePages() {
                const pages = [];
                const start = Math.max(1, this.currentPage - 2);
                const end = Math.min(this.totalPages, this.currentPage + 2);
                for (let i = start; i <= end; i++) pages.push(i);
                return pages;
            },

            TYPE_CONFIG: {
                it_fault: { icon: 'fa-network-wired', label: 'IT故障', color: '#0984e3' },
                leave: { icon: 'fa-umbrella-beach', label: '请假', color: '#f39c12' },
                expense: { icon: 'fa-receipt', label: '报销', color: '#00b894' },
                admin: { icon: 'fa-box', label: '行政', color: '#6c5ce7' },
            },
            STATUS_LABELS: {
                created: '待处理', assigned: '已派发',
                processing: '处理中', resolved: '已解决', closed: '已关闭',
            },

            getTypeIcon(type) { return this.TYPE_CONFIG[type]?.icon || 'fa-file-alt'; },
            getTypeLabel(type) { return this.TYPE_CONFIG[type]?.label || type; },
            getStatusLabel(status) { return this.STATUS_LABELS[status] || status; },
            formatDate(d) { return window.formatDate ? window.formatDate(d) : (d || ''); },

            showToast(message, type) {
                this.toast = { show: true, message, type };
                setTimeout(() => { this.toast = { show: false, message: '', type: 'success' }; }, 3000);
            },
        };
    });
});
