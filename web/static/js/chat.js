/**
 * chat.js — 聊天模块 (SSE 流式、Agent 面板、确认卡片、ReAct 思维链)
 * 依赖: app.js (currentIdentity, getThreadId, getStoredMessages, saveStoredMessages, appendToMsgLog, escapeHtml, showToast)
 */
(function() {
    // DOM refs (init once on first use)
    let _domReady = false;
    function dom() {
        if (!_domReady) {
            window.chatArea = document.getElementById('chat-area');
            window.userInput = document.getElementById('user-input');
            window.sendBtn = document.getElementById('send-btn');
            window.welcome = document.getElementById('welcome');
            window.debugPanel = document.getElementById('debug-panel');
            window.debugContent = document.getElementById('debug-content');
            _domReady = true;
        }
        return { chatArea, userInput, sendBtn, welcome, debugPanel, debugContent };
    }

    // Legacy compat for existing code
    let currentUser = { get user_name() { return window.currentIdentity || '张三'; },
                        get role() { return window.getRole ? window.getRole(window.currentIdentity) : 'employee'; } };
    let threadId = window.threadId || 'web_default';

    // Shared mutable state for streaming (declared in IIFE scope)
    let debugEntries = [];
    let currentBotBubble = null;
    let currentOrchBlock = null;
    let currentOrchSteps = null;
    let currentBotText = '';
    let currentReactPanel = null;
    let currentReactSteps = null;

    // Agent Roster (loaded from API)
    // ============================================================
    const AGENT_ICONS = {
        'it_consultant': { icon: 'fa-network-wired', cls: 'it' },
        'hr_consultant': { icon: 'fa-user-tie', cls: 'hr' },
        'facilities': { icon: 'fa-building', cls: 'analytics' },
    };

    async function loadAgentRoster() {
        try {
            const res = await fetch('/api/agents/list');
            const agents = await res.json();
            const roster = document.getElementById('agent-roster');
            roster.innerHTML = agents.map(a => {
                const info = AGENT_ICONS[a.agent_id] || { icon: 'fa-robot', cls: '' };
                return `
                <div class="agent-card" data-agent="${a.agent_id}">
                    <div class="agent-card-header">
                        <div class="agent-icon ${info.cls}"><i class="fas ${info.icon}"></i></div>
                        <span class="agent-name">${a.name}</span>
                        <span class="agent-status idle" id="status-${a.agent_id}"></span>
                    </div>
                    <div class="agent-desc">${a.description.substring(0, 80)}...</div>
                    <div class="agent-caps">
                        ${a.capabilities.slice(0, 4).map(c => `<span class="agent-cap">${c}</span>`).join('')}
                    </div>
                </div>`;
            }).join('');
        } catch (e) {
            console.error('Failed to load agent roster:', e);
        }
    }

    function setAgentStatus(agentId, status) {
        const el = document.getElementById('status-' + agentId);
        if (el) { el.className = 'agent-status ' + status; }

        // Highlight card
        document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('active'));
        if (status === 'active') {
            const card = document.querySelector(`[data-agent="${agentId}"]`);
            if (card) card.classList.add('active');
        }
    }

    function resetAllAgentStatuses() {
        document.querySelectorAll('.agent-status').forEach(s => {
            s.className = 'agent-status idle';
        });
        document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('active'));
    }

    // ============================================================
    // Pipeline Visualization — Hub & Spoke
    // ============================================================
    function setPipelineTrack(track) {
        const steps = document.querySelectorAll('.pipeline-step');
        steps.forEach(s => {
            s.classList.remove('current', 'done');
            if (s.dataset.track === track) s.classList.add('current');
        });
        // Mark "route" as done once we know the track
        const routeStep = document.querySelector('[data-track="route"]');
        if (routeStep && track) routeStep.classList.add('done');
    }

    function resetPipeline() {
        document.querySelectorAll('.pipeline-step').forEach(s => {
            s.classList.remove('current', 'done');
        });
    }

    // ============================================================
    // Debug Panel
    // ============================================================
    function addDebugEntry(type, agentId, message) {
        const now = new Date().toLocaleTimeString();
        const entry = { ts: now, type, agentId, message };
        debugEntries.push(entry);

        const cls = type === 'error' ? 'error' : type === 'event' ? 'event' : 'agent';
        const div = document.createElement('div');
        div.className = 'debug-entry';
        div.innerHTML = `<span class="ts">${now}</span> <span class="${cls}">[${agentId}]</span> ${message}`;
        debugContent.appendChild(div);
        debugPanel.scrollTop = debugPanel.scrollHeight;
    }

    function toggleDebug() {
        const open = debugPanel.classList.toggle('open');
        document.getElementById('debug-btn').innerHTML = open
            ? '<i class="fas fa-times"></i> 关闭'
            : '<i class="fas fa-bug"></i> 调试';
    }

    // ============================================================
    // Message Handling
    // ============================================================
    // ── sessionStorage 消息持久化 (跨页面导航快速恢复) ──
    const MSG_STORE_KEY = 'service_desk_messages';

    function getStoredMessages() {
        try {
            const raw = sessionStorage.getItem(MSG_STORE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            return [];
        }
    }

    function saveStoredMessages(msgs) {
        try {
            // 最多保留 50 条，防止爆 sessionStorage
            const trimmed = msgs.slice(-50);
            sessionStorage.setItem(MSG_STORE_KEY, JSON.stringify(trimmed));
        } catch (e) {
            // sessionStorage 满了就清旧留新
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

    function hideWelcome() {
        if (welcome) welcome.style.display = 'none';
    }

    function addUserMessage(text) {
        hideWelcome();
        const div = document.createElement('div');
        div.className = 'message user';
        div.innerHTML = `
            <div class="msg-content">
                <div class="msg-bubble">${escapeHtml(text)}</div>
            </div>
            <div class="msg-avatar user"><i class="fas fa-user"></i></div>`;
        chatArea.appendChild(div);
        appendToMsgLog({ role: 'user', content: text });
        scrollToBottom();
    }

    function addBotMessage() {
        hideWelcome();
        // 如果有正在进行的 bot 消息，先保存
        if (currentBotText.trim()) {
            appendToMsgLog({ role: 'assistant', content: currentBotText.trim() });
        }
        currentBotText = '';
        const div = document.createElement('div');
        div.className = 'message bot';
        const bubble = document.createElement('div');
        bubble.className = 'msg-bubble';
        div.innerHTML = '<div class="msg-avatar bot"><i class="fas fa-robot"></i></div><div class="msg-content"></div>';
        div.querySelector('.msg-content').appendChild(bubble);
        chatArea.appendChild(div);
        currentBotBubble = bubble;
        currentOrchBlock = null;
        currentOrchSteps = null;
        return div;
    }

    function ensureOrchBlock() {
        if (!currentBotBubble) return;
        if (!currentOrchBlock) {
            currentOrchBlock = document.createElement('div');
            currentOrchBlock.className = 'orch-block';
            currentOrchBlock.innerHTML = `
                <div class="orch-header" onclick="toggleOrch(this)">
                    <i class="fas fa-chevron-down"></i> 编排过程
                </div>
                <div class="orch-steps"></div>`;
            currentBotBubble.appendChild(currentOrchBlock);
            currentOrchSteps = currentOrchBlock.querySelector('.orch-steps');
        }
    }

    function addOrchStep(iconClass, text) {
        ensureOrchBlock();
        if (!currentOrchSteps) return;
        const step = document.createElement('div');
        step.className = 'orch-step';
        step.innerHTML = `
            <div class="orch-step-icon ${iconClass}"><i class="fas fa-check"></i></div>
            <span class="orch-step-text">${text}</span>`;
        currentOrchSteps.appendChild(step);
    }

    function toggleOrch(header) {
        header.classList.toggle('collapsed');
        const steps = header.nextElementSibling;
        if (steps) steps.style.display = steps.style.display === 'none' ? '' : 'none';
    }

    // ============================================================
    // ReAct Thinking Chain Panel (防坑2: 滚动锁定 + 完成后自动折叠)
    // ============================================================

    // v12: React 步骤计数器
    let reactStepCount = 0;
    let reactToolCallCount = 0;

    function ensureReactPanel() {
        if (!currentBotBubble) return;
        if (!currentReactPanel) {
            reactStepCount = 0;
            reactToolCallCount = 0;
            currentReactPanel = document.createElement('div');
            currentReactPanel.className = 'react-panel';
            currentReactPanel.innerHTML = `
                <div class="react-header" onclick="toggleReact(this)">
                    <i class="fas fa-chevron-down"></i> 🧠 思考过程
                    <span class="react-step-badge" style="display:none;"></span>
                </div>
                <div class="react-steps"></div>
                <div class="react-footer" style="display:none;"></div>`;
            // Insert BEFORE any orch-block or text nodes
            const firstChild = currentBotBubble.firstChild;
            if (firstChild) {
                currentBotBubble.insertBefore(currentReactPanel, firstChild);
            } else {
                currentBotBubble.appendChild(currentReactPanel);
            }
            currentReactSteps = currentReactPanel.querySelector('.react-steps');
        }
    }

    function addReactStep(data) {
        ensureReactPanel();
        if (!currentReactSteps) return;

        reactStepCount++;
        const eventType = data.event || '';
        let icon, text, cls;

        // v12: 增强图标和颜色编码
        if (eventType === 'thought') {
            icon = '💭'; text = data.text || ''; cls = 'thought';
        } else if (eventType === 'tool_call') {
            reactToolCallCount++;
            icon = '🔧'; text = data.text || `调用 ${data.tool || '?'}`; cls = 'tool_call';
        } else if (eventType === 'tool_result') {
            // 判断成功/失败
            var isError = data.text && (data.text.indexOf('❌') === 0 || data.text.indexOf('失败') >= 0);
            icon = isError ? '❌' : '✅';
            text = data.text || ''; cls = isError ? 'tool_error' : 'tool_result';
        } else if (eventType === 'final') {
            icon = '✨'; text = data.text || ''; cls = 'final';
            // 更新 header 显示迭代次数
            var badge = currentReactPanel.querySelector('.react-step-badge');
            if (badge && data.iterations) {
                badge.style.display = '';
                badge.textContent = data.iterations + '轮 · ' + reactToolCallCount + '工具';
            }
        } else {
            icon = '·'; text = JSON.stringify(data); cls = '';
        }

        var step = document.createElement('div');
        step.className = 'react-step ' + cls;
        step.innerHTML = '<span class="react-step-num">' + reactStepCount + '</span>' +
            '<span class="react-icon">' + icon + '</span>' +
            '<span class="react-text">' + escapeHtml(text) + '</span>';
        currentReactSteps.appendChild(step);

        scrollToBottom();
    }

    function toggleReact(header) {
        header.classList.toggle('collapsed');
        var steps = header.nextElementSibling;
        if (steps && steps.classList.contains('react-steps')) {
            steps.style.display = steps.style.display === 'none' ? '' : 'none';
        }
        var footer = currentReactPanel ? currentReactPanel.querySelector('.react-footer') : null;
        if (footer) {
            footer.style.display = footer.style.display === 'none' ? '' : 'none';
        }
    }

    function collapseReactPanel() {
        // v12: 完成后不隐藏面板，改为折叠+显示摘要
        if (currentReactPanel) {
            var header = currentReactPanel.querySelector('.react-header');
            var footer = currentReactPanel.querySelector('.react-footer');
            var steps = currentReactPanel.querySelector('.react-steps');

            // 显示摘要 footer
            if (footer) {
                footer.style.display = '';
                footer.innerHTML = '<i class="fas fa-check-circle" style="color:var(--success);"></i> ' +
                    '共 <b>' + reactStepCount + '</b> 步推理 · ' +
                    '<b>' + reactToolCallCount + '</b> 个工具调用 · 完成';
            }

            // 折叠 (但保持 header 可见)
            if (header && !header.classList.contains('collapsed')) {
                header.classList.add('collapsed');
            }
            if (steps) steps.style.display = 'none';
            if (footer) footer.style.display = '';

            // 重置计数器
            reactStepCount = 0;
            reactToolCallCount = 0;
        }
    }

    function addBotText(text) {
        if (!currentBotBubble) addBotMessage();
        currentBotText += text;
        // Append text node or replace
        const existing = currentBotBubble.querySelector('.bot-text');
        if (existing) {
            existing.textContent += text;
        } else {
            const span = document.createElement('span');
            span.className = 'bot-text';
            span.textContent = text;
            currentBotBubble.insertBefore(span, currentBotBubble.firstChild);
        }
        scrollToBottom();
    }

    function showThinking(text) {
        hideWelcome();
        const div = document.createElement('div');
        div.className = 'thinking';
        div.id = 'thinking-indicator';
        div.innerHTML = `
            <div class="thinking-dots"><span></span><span></span><span></span></div>
            <span class="thinking-text">${escapeHtml(text || 'AI Agent 思考中...')}</span>`;
        chatArea.appendChild(div);
        scrollToBottom();
    }

    function updateThinking(text) {
        const el = document.getElementById('thinking-indicator');
        if (el) {
            const textEl = el.querySelector('.thinking-text');
            if (textEl) textEl.textContent = text;
        } else {
            showThinking(text);
        }
        scrollToBottom();
    }

    function hideThinking() {
        const el = document.getElementById('thinking-indicator');
        if (el) {
            el.style.opacity = '0';
            el.style.transition = 'opacity 0.25s';
            setTimeout(() => el.remove(), 250);
        }
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ============================================================
    // Confirmation Card Rendering
    // ============================================================

    function renderConfirmCard(cardData) {
        if (!cardData || !cardData.type) return null;

        const wrapper = document.createElement('div');
        wrapper.className = 'confirm-card';
        wrapper.setAttribute('data-card', JSON.stringify(cardData));

        // Title
        const title = document.createElement('div');
        title.className = 'card-title';
        title.textContent = cardData.title || '确认信息';
        wrapper.appendChild(title);

        // Alerts (conflict warnings, suggestions)
        if (cardData.alerts && cardData.alerts.length > 0) {
            cardData.alerts.forEach(alert => {
                const alertEl = document.createElement('div');
                alertEl.className = 'card-alert ' + (alert.type || 'info');
                alertEl.textContent = alert.message;
                wrapper.appendChild(alertEl);
            });
        }

        // Description (support newlines)
        if (cardData.description) {
            const desc = document.createElement('div');
            desc.className = 'card-desc';
            desc.innerHTML = cardData.description.replace(/\n/g, '<br>');
            wrapper.appendChild(desc);
        }

        // Form fields
        const fields = cardData.fields || [];
        const formData = {};
        fields.forEach(field => {
            const fg = document.createElement('div');
            fg.className = 'card-field';

            const label = document.createElement('label');
            label.textContent = (field.required ? '* ' : '') + (field.label || field.key);
            fg.appendChild(label);

            let input;
            if (field.type === 'select' && field.options) {
                input = document.createElement('select');
                field.options.forEach(opt => {
                    const o = document.createElement('option');
                    o.value = opt.value;
                    o.textContent = opt.label;
                    if (opt.value === field.value) o.selected = true;
                    input.appendChild(o);
                });
            } else if (field.type === 'date') {
                input = document.createElement('input');
                input.type = 'date';
                input.value = field.value || '';
            } else if (field.type === 'number') {
                input = document.createElement('input');
                input.type = 'number';
                input.value = field.value || '';
                input.min = field.min || 0;
                if (field.max) input.max = field.max;
            } else {
                input = document.createElement('input');
                input.type = 'text';
                input.value = field.value || '';
                if (field.placeholder) input.placeholder = field.placeholder;
            }
            input.setAttribute('data-field-key', field.key);
            input.required = field.required || false;
            fg.appendChild(input);

            // Field hint
            if (field.hint) {
                const hint = document.createElement('div');
                hint.className = 'card-field-hint';
                hint.textContent = field.hint;
                fg.appendChild(hint);
            }

            formData[field.key] = input;
            wrapper.appendChild(fg);
        });

        // Action buttons
        const actions = document.createElement('div');
        actions.className = 'card-actions';

        const dismissText = cardData.dismiss_text || '取消';
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn';
        cancelBtn.textContent = dismissText;
        cancelBtn.onclick = () => {
            // v8: 动态卡片通过 /chat/resume 取消, 旧卡片通过 /chat/reset-card
            if (cardData.confirm_action === 'chat') {
                fetch('/chat/resume', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        thread_id: window.threadId || 'web_default',
                        action: 'cancel',
                    }),
                }).catch(() => {});
            } else {
                // 清除后端卡片锁状态
                fetch('/chat/reset-card', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        message: '',
                        thread_id: window.threadId || 'web_default',
                        user_name: window.currentIdentity,
                        role: (window.getRole ? window.getRole(window.currentIdentity) : 'employee'),
                    }),
                }).catch(() => {});
            }
            wrapper.innerHTML = '<div class="card-fallback"><span style="color:var(--text-secondary);">' +
                (dismissText === '问题已解决' ? '✅ 问题已解决，无需创建工单' : '已取消') +
                '</span></div>';
        };
        actions.appendChild(cancelBtn);

        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'btn primary';
        confirmBtn.textContent = cardData.confirm_text || '确认';
        confirmBtn.onclick = async () => {
            // Collect form values
            const values = {};
            let valid = true;
            fields.forEach(field => {
                const el = formData[field.key];
                const val = el.value.trim();
                if (field.required && !val) {
                    el.style.borderColor = 'var(--danger)';
                    valid = false;
                } else {
                    el.style.borderColor = 'var(--border)';
                }
                values[field.key] = val;
            });
            if (!valid) return;

            // ── v8: LangGraph interrupt() 恢复 (DynamicActionAgent) ──
            if (cardData.confirm_action === 'chat') {
                // 防坑3: 禁用同气泡内所有卡片按钮，防止并发竞态
                const allCards = wrapper.closest('.msg-content')?.querySelectorAll('.confirm-card');
                allCards?.forEach(c => {
                    c.querySelectorAll('button').forEach(b => b.disabled = true);
                });

                confirmBtn.disabled = true;
                confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                cancelBtn.disabled = true;

                try {
                    const res = await fetch('/chat/resume', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            thread_id: window.threadId || 'web_default',
                            action: 'confirm',
                        }),
                    });

                    if (res.ok) {
                        const data = await res.json();
                        if (data.status === 'ok') {
                            // v10: 所有卡片已确认 → 显示 LLM 最终回复到聊天区域
                            const finalMsg = data.message || cardData.success_message || '操作成功！';
                            // 把最终回复追加为新 bot 消息
                            if (finalMsg && finalMsg !== '操作成功！') {
                                addBotMessage();
                                if (currentBotBubble) {
                                    currentBotBubble.innerHTML = escapeHtml(finalMsg).replace(/\n/g, '<br>');
                                }
                            }
                            // 卡片区显示简洁成功提示
                            wrapper.innerHTML = `
                                <div class="card-success">
                                    <i class="fas fa-check-circle"></i>
                                    <p><strong>✅ 操作已完成</strong></p>
                                </div>`;
                        } else if (data.status === 'interrupted') {
                            // v10: 后续工单新卡片 — 原地替换为新产品卡片
                            const newCards = data.cards || [];
                            if (newCards.length > 0) {
                                wrapper.innerHTML = '';
                                newCards.forEach(c => {
                                    const newCardEl = renderConfirmCard(c);
                                    if (newCardEl) wrapper.appendChild(newCardEl);
                                });
                            } else {
                                // 无新卡片但有消息 → 追加为 bot 消息
                                if (data.message && data.message.trim()) {
                                    addBotMessage();
                                    if (currentBotBubble) {
                                        currentBotBubble.innerHTML = escapeHtml(data.message).replace(/\n/g, '<br>');
                                    }
                                }
                                wrapper.innerHTML = '<div style="padding:8px;color:var(--text-secondary);">⏳ 继续处理中...</div>';
                            }
                        } else {
                            wrapper.innerHTML = `<div class="card-fallback" style="color:var(--danger);padding:12px;">
                                ❌ ${escapeHtml(data.message || '操作失败')}
                            </div>`;
                        }
                    } else {
                        const err = await res.json().catch(() => ({}));
                        wrapper.innerHTML = `<div class="card-fallback" style="color:var(--danger);padding:12px;">
                            ❌ ${err.detail || '操作失败，请重试'}
                        </div>`;
                    }
                } catch (e) {
                    console.error('Resume error:', e);
                    wrapper.innerHTML = `<div class="card-fallback" style="color:var(--danger);padding:12px;">
                        ❌ 请求失败：${e.message}
                    </div>`;
                }
                return;
            }

            // Build request (旧卡片路径: REST API 直接调用)
            let actionUrl = cardData.action || '';
            for (const [k, v] of Object.entries(values)) {
                actionUrl = actionUrl.replace(`{${k}}`, encodeURIComponent(v));
            }

            const method = cardData.method || 'POST';
            let body;
            if (cardData.body_template) {
                body = { ...cardData.body_template };
                for (const [k, v] of Object.entries(values)) {
                    body[k] = v;
                }
                body.skip_card = true;  // 卡片确认模式：直接创建工单
                // ★ 注入当前用户身份 — 确保工单关联到正确用户
                body.user_id = body.user_id || window.currentIdentity || '';
                body.user_name = body.user_name || window.currentIdentity || '';
            } else {
                if (actionUrl.includes('/book')) {
                    body = {
                        date: values.date || new Date().toISOString().split('T')[0],
                        start_time: values.start_time || '14:00',
                        end_time: values.end_time || '15:00',
                        title: values.title || '会议',
                        booked_by: values.booked_by || window.currentIdentity || 'web_user',
                        description: values.description || '',
                    };
                } else {
                    body = values;
                }
                // 其他路径也注入身份
                body.user_id = body.user_id || window.currentIdentity || '';
                body.user_name = body.user_name || window.currentIdentity || '';
            }

            // Show loading
            confirmBtn.disabled = true;
            confirmBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 处理中...';
            cancelBtn.disabled = true;

            try {
                const res = await fetch(actionUrl, {
                    method: method,
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                });

                if (res.ok) {
                    const data = await res.json();
                    const fallbackHtml = cardData.fallback_url ? `
                        <div class="card-fallback">
                            <a href="${cardData.fallback_url}">
                                <i class="fas fa-arrow-right"></i> ${cardData.fallback_text || '查看详情'}
                            </a>
                        </div>` : '';
                    wrapper.innerHTML = `
                        <div class="card-success">
                            <i class="fas fa-check-circle"></i>
                            <p><strong>${cardData.success_message || '操作成功！'}</strong></p>
                        </div>
                        ${fallbackHtml}
                    `;
                } else {
                    const err = await res.json();
                    confirmBtn.disabled = false;
                    cancelBtn.disabled = false;
                    confirmBtn.textContent = cardData.confirm_text || '确认';
                    confirmBtn.innerHTML = cardData.confirm_text || '确认';
                    wrapper.innerHTML = `<div class="card-fallback" style="color:var(--danger);padding:12px;">
                        ❌ ${err.detail || '操作失败，请重试'}
                    </div>`;
                }
            } catch (e) {
                console.error('Card submit error:', e);
                wrapper.innerHTML = `<div class="card-fallback" style="color:var(--danger);padding:12px;">
                    ❌ 请求失败：${e.message}
                </div>`;
            }
        };
        actions.appendChild(confirmBtn);
        wrapper.appendChild(actions);

        // Fallback link
        if (cardData.fallback_url) {
            const fallback = document.createElement('div');
            fallback.className = 'card-fallback';
            fallback.innerHTML = `<a href="${cardData.fallback_url}" target="_blank">
                <i class="fas fa-external-link-alt"></i> ${cardData.fallback_text || '在新页面打开'}
            </a>`;
            wrapper.appendChild(fallback);
        }

        return wrapper;
    }

    // ============================================================
    // Streaming Chat — Hub & Spoke with real-time token streaming
    // ============================================================

    async function sendMessage(text) {
        // 防御：确保 DOM 引用已初始化
        if (!_domReady) dom();
        const inputEl = userInput || window.userInput;
        const btnEl = sendBtn || window.sendBtn;

        if (!inputEl) {
            console.error('[sendMessage] userInput element not found');
            return;
        }

        const input = text || inputEl.value.trim();
        if (!input) return;

        console.log('[sendMessage] Sending:', input.substring(0, 80));

        try {
            inputEl.value = '';
            if (btnEl) {
                btnEl.disabled = true;
                btnEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 思考中...';
            }

            addUserMessage(input);
            resetPipeline();
            resetAllAgentStatuses();
            addDebugEntry('event', 'system', `输入: ${input.substring(0, 80)}`);

            // Show simple thinking dots
            showThinking('AI Agent 思考中...');

            let botBubbleCreated = false;
            let currentTrack = '';

            const res = await fetch('/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: input,
                    thread_id: window.threadId || 'web_default',
                    user_name: window.currentIdentity,
                    role: (window.getRole ? window.getRole(window.currentIdentity) : 'employee'),
                }),
            });

            if (!res.ok) {
                const errText = await res.text().catch(() => '');
                throw new Error(`服务异常 (${res.status}): ${errText.substring(0, 200)}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let rawChunks = 0;
            let totalBytes = 0;
            console.log('[sendMessage] Stream opened, reading...');

            // ── 内联 token 处理函数（减少重复代码）──
            const processToken = (trimmed) => {
                if (!trimmed) return;

                if (trimmed.startsWith('[REACT]')) {
                    const reactJson = trimmed.substring(7);
                    try {
                        const reactData = JSON.parse(reactJson);
                        if (!botBubbleCreated) {
                            hideThinking();
                            addBotMessage();
                            botBubbleCreated = true;
                        }
                        ensureReactPanel();
                        addReactStep(reactData);
                    } catch (e) {
                        console.error('REACT parse error:', e);
                    }
                } else if (trimmed.startsWith('[STREAM]')) {
                    const content = trimmed.substring(8);
                    if (!botBubbleCreated) {
                        hideThinking();
                        addBotMessage();
                        botBubbleCreated = true;
                    }
                    addBotText(content);
                } else if (trimmed.startsWith('[THINKING]')) {
                    const content = trimmed.replace('[THINKING]', '').trim();
                    updateThinking(content);
                    addDebugEntry('event', 'thinking', content);
                } else if (trimmed.startsWith('[PROGRESS]')) {
                    const content = trimmed.replace('[PROGRESS]', '').trim();
                    updateThinking(content);
                    addDebugEntry('event', 'progress', content);
                } else if (trimmed.startsWith('[ROUTE]')) {
                    const content = trimmed.replace('[ROUTE]', '').trim();
                    if (content.includes('极速')) currentTrack = 'fast';
                    else if (content.includes('动作')) currentTrack = 'action';
                    else if (content.includes('复杂')) currentTrack = 'complex';
                    else if (content.includes('兜底')) currentTrack = 'fallback';
                    setPipelineTrack(currentTrack);
                    addOrchStep('classify', `路由: ${content}`);
                    addDebugEntry('event', 'router', content);
                } else if (trimmed.startsWith('[FAST]')) {
                    currentTrack = 'fast';
                    setPipelineTrack('fast');
                    addOrchStep('classify', trimmed.replace('[FAST]', '').trim());
                    addDebugEntry('event', 'fast_track', '进入极速通道');
                } else if (trimmed.startsWith('[ACTION_QUERY]')) {
                    currentTrack = 'action_query';
                    setPipelineTrack('action_query');
                    addOrchStep('query', trimmed.replace('[ACTION_QUERY]', '').trim());
                    addDebugEntry('event', 'action_query_track', '进入数据查询通道');
                } else if (trimmed.startsWith('[ACTION]')) {
                    currentTrack = 'action';
                    setPipelineTrack('action');
                    addOrchStep('delegate', trimmed.replace('[ACTION]', '').trim());
                    addDebugEntry('event', 'action_track', '进入动作通道');
                } else if (trimmed.startsWith('[COMPLEX]')) {
                    currentTrack = 'complex';
                    setPipelineTrack('complex');
                    addOrchStep('plan', trimmed.replace('[COMPLEX]', '').trim());
                    addDebugEntry('event', 'complex_track', '进入复杂通道');
                } else if (trimmed.startsWith('[CLARIFY]')) {
                    // AI 反问澄清 — 不显示标签文字，仅记录
                    addDebugEntry('event', 'clarify', 'AI 反问澄清意图');
                } else if (trimmed.startsWith('[INTERRUPT_CARD]')) {
                    // 卡片中断 — 流式标签仅用于管道可视化
                    addDebugEntry('event', 'interrupt_card', '等待卡片确认');
                } else if (trimmed.startsWith('[DYNAMIC]')) {
                    currentTrack = 'dynamic';
                    setPipelineTrack('dynamic');
                    const dynContent = trimmed.replace('[DYNAMIC]', '').trim();
                    updateThinking(dynContent);
                    addDebugEntry('event', 'dynamic_track', dynContent);
                } else if (trimmed.startsWith('[CARD]')) {
                    const cardJson = trimmed.substring(6);
                    try {
                        const cardData = JSON.parse(cardJson);
                        console.log('[sendMessage] Card parsed:', cardData.type, cardData.title);
                        hideThinking();
                        if (!botBubbleCreated) {
                            addBotMessage();
                            botBubbleCreated = true;
                        }
                        const cardEl = renderConfirmCard(cardData);
                        if (cardEl && currentBotBubble) {
                            currentBotBubble.appendChild(cardEl);
                            scrollToBottom();
                        } else {
                            console.warn('[sendMessage] Card render returned null. cardData:', cardData);
                        }
                        addDebugEntry('event', 'card', '确认卡片: ' + (cardData.type || 'unknown'));
                    } catch (e) {
                        console.error('[sendMessage] Card parse error:', e);
                        console.error('[sendMessage] Raw card JSON (first 200):', cardJson.substring(0, 200));
                    }
                } else if (trimmed.startsWith('[FALLBACK]')) {
                    currentTrack = 'fallback';
                    setPipelineTrack('fallback');
                    addOrchStep('classify', trimmed.replace('[FALLBACK]', '').trim());
                    addDebugEntry('event', 'fallback', '兜底模式');
                } else if (trimmed.startsWith('[CARD_RESPONSE]')) {
                    // 卡片回复已处理 — 仅记录，不显示标签文字
                    addDebugEntry('event', 'card_response', '卡片回复已处理');
                } else if (trimmed.startsWith('[RESPONSE]')) {
                    // 响应节点 — 静默，后续 [STREAM] 携带实际内容
                    addDebugEntry('event', 'respond', '响应完成');
                } else if (trimmed === '[INTERRUPT]') {
                    addDebugEntry('event', 'system', '图已冻结 — 等待用户确认');
                    updateThinking('⏸️ 等待确认卡片...');
                } else if (trimmed === '[DONE]') {
                    addDebugEntry('event', 'system', '编排完成');
                    collapseReactPanel();
                } else if (trimmed.startsWith('[ORCHESTRATOR]')) {
                    const content = trimmed.replace('[ORCHESTRATOR]', '').trim();
                    updateThinking(content);
                    addOrchStep('classify', content);
                    addDebugEntry('event', 'orchestrator', content);
                } else if (trimmed.startsWith('[AGENT:')) {
                    const match = trimmed.match(/\[AGENT:([^\]]+)\]\s*(.*)/);
                    if (match) {
                        const agentId = match[1], msg = match[2];
                        setAgentStatus(agentId, 'active');
                        addOrchStep('delegate', `${agentId}: ${msg}`);
                        addDebugEntry('agent', agentId, msg);
                        updateThinking(`${agentId}: ${msg}`);
                    }
                } else if (trimmed.startsWith('[DEBUG]')) {
                    addDebugEntry('event', 'debug', trimmed.replace('[DEBUG]', '').trim());
                } else if (trimmed.startsWith('[ERROR]')) {
                    const content = trimmed.replace('[ERROR]', '').trim();
                    addDebugEntry('error', 'system', content);
                    if (!botBubbleCreated) {
                        hideThinking();
                        addBotMessage();
                        botBubbleCreated = true;
                    }
                    addBotText('❌ ' + content);
                } else if (trimmed.startsWith('[RESPONSE]')) {
                    hideThinking();
                    if (!botBubbleCreated) {
                        addBotMessage();
                        botBubbleCreated = true;
                    }
                } else {
                    // Unmarked text — treat as response content
                    if (!botBubbleCreated) {
                        hideThinking();
                        addBotMessage();
                        botBubbleCreated = true;
                    }
                    addBotText(trimmed + '\n');
                }
            };

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    console.log('[sendMessage] Stream ended. Chunks:', rawChunks, 'Bytes:', totalBytes);
                    break;
                }

                rawChunks++;
                totalBytes += value ? value.length : 0;
                const chunkText = decoder.decode(value, { stream: true });
                if (rawChunks <= 3) {
                    console.log('[sendMessage] Chunk', rawChunks, ':', JSON.stringify(chunkText.substring(0, 100)));
                }

                buffer += chunkText;
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    processToken(line.trim());
                }
            }

            // ★ 处理流结束后 buffer 中剩余的最后一行
            if (buffer.trim()) {
                processToken(buffer.trim());
            }

            // Finalize — 保存 bot 消息到 sessionStorage
            hideThinking();
            resetAllAgentStatuses();
            if (currentBotText.trim()) {
                appendToMsgLog({ role: 'assistant', content: currentBotText.trim() });
                currentBotText = '';
            }

            // 诊断：流结束但无 bot 回复 → 显示可见错误
            if (!botBubbleCreated && rawChunks === 0) {
                console.warn('[sendMessage] Empty stream — no chunks received');
                addBotMessage();
                if (currentBotBubble) {
                    currentBotBubble.innerHTML = '<span style="color:#e17055;">⚠️ 服务返回了空响应，请检查后端日志或重启服务后刷新页面 (Ctrl+Shift+R)。</span>';
                }
                addDebugEntry('error', 'system', '空响应：流无数据');
            }

        } catch (error) {
            hideThinking();
            console.error('Chat error:', error);
            if (currentBotBubble) {
                currentBotBubble.innerHTML = '<span style="color:#e17055;">抱歉，服务暂时不可用，请稍后重试。</span>';
            } else {
                addBotMessage();
                if (currentBotBubble) {
                    currentBotBubble.innerHTML = '<span style="color:#e17055;">抱歉，服务暂时不可用，请稍后重试。</span>';
                }
            }
            addDebugEntry('error', 'system', error.message);
        } finally {
            if (btnEl) {
                btnEl.disabled = false;
                btnEl.innerHTML = '<i class="fas fa-paper-plane"></i> 发送';
            }
            if (inputEl) inputEl.focus();

            if (currentBotBubble) currentBotBubble.removeAttribute('id');
            currentBotBubble = null;
            currentOrchBlock = null;
            currentOrchSteps = null;
            currentReactPanel = null;
            currentReactSteps = null;
            currentBotText = '';
        }
    }

    function sendQuickAction(text) {
        const inputEl = userInput || window.userInput;
        if (inputEl) inputEl.value = text;
        sendMessage(text);
    }

    function clearChat() {
        const messages = chatArea.querySelectorAll('.message, .thinking');
        messages.forEach(m => m.remove());
        if (welcome) welcome.style.display = '';
        resetPipeline();
        resetAllAgentStatuses();
        debugEntries = [];
        debugContent.innerHTML = '<div class="debug-entry" style="color:#6c7086;">已清空</div>';
        currentBotBubble = null;
        currentOrchBlock = null;
        currentOrchSteps = null;
        currentBotText = '';
        // 清除 sessionStorage 消息缓存
        try { sessionStorage.removeItem(window.MSG_STORE_KEY || 'service_desk_messages'); } catch (e) {}
        // 生成新 thread_id，开始全新会话
        threadId = 'web_' + Date.now();
        localStorage.setItem(window.THREAD_KEY || 'service_desk_thread_id', threadId);
        window.threadId = threadId;
        // 通知后端重置
        fetch('/chat/reset', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message: '', thread_id: threadId, user_name: window.currentIdentity }),
        }).catch(() => {});
        userInput.focus();
    }

    function toggleSidebar() {
        document.getElementById('sidebar').classList.toggle('open');
    }

    // ============================================================
    // Chat History Recovery — sessionStorage 快速恢复 + 后端兜底
    // ============================================================
    function renderMsgDOM(msg) {
        if (msg.role === 'user') {
            const div = document.createElement('div');
            div.className = 'message user';
            div.innerHTML = `
                <div class="msg-content">
                    <div class="msg-bubble">${escapeHtml(msg.content)}</div>
                </div>
                <div class="msg-avatar user"><i class="fas fa-user"></i></div>`;
            chatArea.appendChild(div);
        } else {
            const div = document.createElement('div');
            div.className = 'message bot';
            div.innerHTML = `
                <div class="msg-avatar bot"><i class="fas fa-robot"></i></div>
                <div class="msg-content">
                    <div class="msg-bubble">${escapeHtml(msg.content).replace(/\n/g, '<br>')}</div>
                </div>`;
            chatArea.appendChild(div);
        }
    }

    async function restoreChatHistory() {
        let messages = [];

        // 1) sessionStorage 快速恢复 (毫秒级)
        const stored = getStoredMessages();
        if (stored.length > 0) {
            messages = stored;
            console.log(`[Restore] sessionStorage: ${messages.length} messages restored`);
        }

        // 2) 后端 LangGraph checkpointer 兜底 (如果 sessionStorage 是空的)
        if (messages.length === 0) {
            try {
                const res = await fetch(`/chat/history?thread_id=${encodeURIComponent(window.threadId || 'web_default')}`);
                if (res.ok) {
                    const data = await res.json();
                    messages = data.messages || [];
                    if (messages.length > 0) {
                        console.log(`[Restore] Backend: ${messages.length} messages restored`);
                        // 同步到 sessionStorage 供下次快速恢复
                        saveStoredMessages(messages);
                    }
                }
            } catch (e) {
                console.error('Failed to restore from backend:', e);
            }
        }

        if (messages.length === 0) return;

        hideWelcome();

        for (const msg of messages) {
            renderMsgDOM(msg);
        }
        scrollToBottom();
    }

    // ============================================================
    // Init DOM refs on load (script runs at bottom of <body>, DOM is ready)
    dom();

    // ============================================================
    // Expose to global
    window.sendMessage = sendMessage;
    window.sendQuickAction = sendQuickAction;
    window.clearChat = clearChat;
    window.loadAgentRoster = loadAgentRoster;
    window.restoreChatHistory = restoreChatHistory;
    window.addDebugEntry = addDebugEntry;
    window.toggleDebug = toggleDebug;
    window.setAgentStatus = setAgentStatus;
    window.resetAllAgentStatuses = resetAllAgentStatuses;
    window.setPipelineTrack = setPipelineTrack;
    window.resetPipeline = resetPipeline;
    window.renderConfirmCard = renderConfirmCard;
})();
