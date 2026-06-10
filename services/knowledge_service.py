# services/knowledge_service.py
# v2 — IndexIDMap 支持真删除 + 反馈机制 + 扩展默认知识库

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from db.db_router import DatabaseRouter
from .text_embedding import embed_input
import logging

logger = logging.getLogger(__name__)


class KnowledgeService:
    """
    知识库服务类 — 向量检索 + 数据库存储 + 反馈

    v2 改进：
    - FAISS IndexIDMap：支持 remove_ids 真删除，不再依赖伪删除+脏重建
    - 反馈机制：记录用户 help/not-help 反馈
    - 扩展默认知识库：覆盖 IT、请假、报销、行政四大领域
    """

    def __init__(self, db_path: str = 'sqlite:///data/ticket_dispatch.db'):
        self.db_router = DatabaseRouter(db_path)
        self.db = self.db_router.knowledge
        self.index = None
        self._index_id_map: dict[int, int] = {}  # doc_id → FAISS 内部位置（兼容）
        self.initialized = False

        # 默认知识库内容 — 覆盖 IT、HR/请假、报销、行政
        self.default_knowledge = self._build_default_knowledge()

    @staticmethod
    def _build_default_knowledge() -> list[dict]:
        """构建扩展默认知识库 — 五大领域，合理分类"""
        return [
            # ================================================================
            # 一、IT 服务台
            # ================================================================
            # -- 服务概览 --
            {
                "content": (
                    "IT服务台概览：工作时间为工作日周一至周五 9:00-18:00，"
                    "紧急故障可通过P0工单通道7×24小时响应。"
                    "总部大楼3层305室，内线电话8888，外线010-8888XXXX。"
                    "常见服务范围：网络故障、账号管理、设备维修、软件安装、系统运维、安全事件。"
                ),
                "category": "IT-服务概览",
                "keywords": ["工作时间", "服务时间", "几点", "周末", "24小时", "紧急", "地址", "电话", "联系方式", "服务台"]
            },
            # -- 网络故障 --
            {
                "content": (
                    "VPN连接失败排查步骤：1. 检查本机网络连通性(ping网关) "
                    "2. 确认VPN客户端版本≥v3.2.1 3. 验证AD账号密码正确性 "
                    "4. 检查防火墙是否放行VPN端口(UDP 1194) 5. 如仍不通，提交P1工单处理。"
                ),
                "category": "IT-网络与连接",
                "keywords": ["VPN", "连接失败", "远程", "网络", "防火墙", "ping"]
            },
            {
                "content": (
                    "企业WiFi网络配置：SSID名为Corp-Net（员工用）和Corp-Guest（访客用）。"
                    "员工连接需使用AD账号密码认证，访客需接待人扫码授权。"
                    "如遇WiFi频繁断连，请先忘记网络后重新连接。"
                ),
                "category": "IT-网络与连接",
                "keywords": ["WiFi", "无线", "上网", "SSID", "断连", "网络连接"]
            },
            # -- 账号管理 --
            {
                "content": (
                    "账号锁定/密码重置流程：1. AD账号连续5次输错自动锁定30分钟 "
                    "2. 员工可通过OA自助解锁 3. 密码重置需提交工单，审批后15分钟内生效 "
                    "4. 新员工账号创建需HR系统流程完成后自动开通。"
                    "密码要求：12位以上，包含大小写字母、数字和特殊符号。"
                ),
                "category": "IT-账号与权限",
                "keywords": ["密码", "账号", "锁定", "重置", "登录", "新员工", "OA", "忘记密码"]
            },
            # -- 系统运维 --
            {
                "content": (
                    "数据库连接超时常见原因：1. 网络策略变更导致端口不通(默认3306/5432) "
                    "2. 数据库最大连接数已满 3. 应用连接池配置不当 "
                    "4. 数据库服务未启动。建议先检查连接字符串和网络连通性。"
                ),
                "category": "IT-系统运维",
                "keywords": ["数据库", "超时", "MySQL", "PostgreSQL", "连接", "连接池"]
            },
            {
                "content": (
                    "服务器重启标准流程：1. 确认重启窗口时间(非工作时间) "
                    "2. 通知受影响的业务团队 3. 备份关键配置和数据 "
                    "4. 逐台重启(集群环境) 5. 重启后验证服务可达性和数据一致性 "
                    "6. 如遇重启失败，升级为P1工单。"
                ),
                "category": "IT-系统运维",
                "keywords": ["重启", "服务器", "维护", "停机", "备份", "窗口"]
            },
            # -- 信息安全 --
            {
                "content": (
                    "信息安全策略要点：所有系统必须使用强密码(12位+大小写+数字+符号)、"
                    "敏感数据传输必须加密(HTTPS/TLS1.2+)、"
                    "生产环境未经审批不得直接访问、"
                    "离职员工账号需在最后工作日24小时内注销。"
                    "发现可疑活动请立即联系IT安全团队。"
                ),
                "category": "IT-信息安全",
                "keywords": ["安全", "密码", "加密", "权限", "审计", "合规", "数据泄露"]
            },

            # ================================================================
            # 二、工单与 SLA
            # ================================================================
            {
                "content": (
                    "工单SLA政策（服务等级协议）：\n"
                    "🔴 P0(紧急)：响应15分钟/解决4小时 — 系统宕机、核心业务中断、多人受影响\n"
                    "🟠 P1(高)：响应1小时/解决8小时 — 影响工作效率但可暂时绕过\n"
                    "🟡 P2(中)：响应4小时/解决24小时 — 一般故障/申请，有替代方案\n"
                    "🟢 P3(低)：响应8小时/解决48小时 — 咨询、非紧急问题\n"
                    "超时未响应自动升级优先级并通知主管。"
                ),
                "category": "服务-SLA政策",
                "keywords": ["SLA", "响应", "解决", "优先级", "超时", "升级", "P0", "P1", "P2", "P3"]
            },

            # ================================================================
            # 三、请假制度（详细的请假规则）
            # ================================================================
            # -- 年假 --
            {
                "content": (
                    "【年假（带薪年休假）详细规则】\n"
                    "一、年假天数\n"
                    "- 工龄1年≤Y<10年：5天/年\n"
                    "- 工龄10年≤Y<20年：10天/年\n"
                    "- 工龄≥20年：15天/年\n"
                    "二、申请规则\n"
                    "- 提前至少3个工作日提交OA请假申请\n"
                    "- 由直属主管审批，连续3天及以上需部门总监加签\n"
                    "- 可分次使用，最小单位0.5天\n"
                    "- 可累计至下年度，但累计不超过应享天数的2倍\n"
                    "三、特殊情况\n"
                    "- 法定节假日、公休日不计入年假天数\n"
                    "- 年假期间遇节假日自动顺延\n"
                    "- 离职时剩余年假按日工资折算补偿"
                ),
                "category": "请假-年假",
                "keywords": ["年假", "带薪休假", "工龄", "审批", "累计", "申请天数", "年休假"]
            },
            # -- 病假 --
            {
                "content": (
                    "【病假规则】\n"
                    "一、申请流程\n"
                    "- 当天上午10:00前通知直属主管并提交病假申请\n"
                    "- OA系统提交→主管审批→HR备案\n"
                    "二、证明材料\n"
                    "- 1-2天：无需医院证明，但需在OA中简述症状\n"
                    "- 3天及以上：需二级甲等以上医院开具的病假证明\n"
                    "- 证明需包含：诊断结果、建议休息天数、医院公章\n"
                    "三、薪资待遇\n"
                    "- 病假工资=基本工资×80%\n"
                    "- 当月病假累计≤2天不扣薪（视同全勤）\n"
                    "- 超过2天部分按80%计薪\n"
                    "四、注意事项\n"
                    "- 虚假病假一经发现按旷工处理\n"
                    "- 长期病假(>15天)需HR总监审批"
                ),
                "category": "请假-病假",
                "keywords": ["病假", "医院证明", "生病", "医生", "病假工资", "病假条", "诊断", "不舒服"]
            },
            # -- 事假 --
            {
                "content": (
                    "【事假规则】\n"
                    "一、申请规则\n"
                    "- 提前1个工作日申请，紧急事假可当天申请但需经理特批\n"
                    "- 单次事假不超过3天，月累计不超过5天\n"
                    "- 事假优先从年假/调休余额中抵扣\n"
                    "二、薪资\n"
                    "- 有年假/调休余额：抵扣带薪假，不扣工资\n"
                    "- 无余额：按无薪事假处理，扣减当日工资\n"
                    "三、审批\n"
                    "- 1天以内：直属主管审批\n"
                    "- 2-3天：主管+部门经理审批\n"
                    "- 紧急事假：需额外说明紧急原因"
                ),
                "category": "请假-事假",
                "keywords": ["事假", "无薪", "经理审批", "特批", "紧急请假", "有事", "私事", "办事"]
            },
            # -- 婚假 --
            {
                "content": (
                    "【婚假规则】\n"
                    "- 法定婚假3天，晚婚（男≥25岁/女≥23岁）15天\n"
                    "- 需提供结婚证复印件至HR\n"
                    "- 需提前2周申请\n"
                    "- 婚假须在结婚登记后6个月内一次性休完\n"
                    "- 婚假为带薪假，薪资照常发放"
                ),
                "category": "请假-婚假",
                "keywords": ["婚假", "结婚", "晚婚", "结婚证", "婚礼"]
            },
            # -- 产假/陪产假 --
            {
                "content": (
                    "【产假/陪产假规则】\n"
                    "一、产假\n"
                    "- 基本产假98天（含产前15天）\n"
                    "- 难产+15天，多胞胎每多1个+15天\n"
                    "- 需提供医院出具的生育证明\n"
                    "二、陪产假\n"
                    "- 15天，需提供配偶生育证明\n"
                    "- 须在配偶产后1个月内休完\n"
                    "三、哺乳假\n"
                    "- 产后1年内，每天1小时哺乳时间（可分两次使用）\n"
                    "四、申请\n"
                    "- 需提前2周提交OA申请\n"
                    "- 所有产假/陪产假均为带薪假"
                ),
                "category": "请假-产假",
                "keywords": ["产假", "陪产假", "生育", "哺乳", "难产", "多胞胎", "怀孕"]
            },
            # -- 调休 --
            {
                "content": (
                    "【调休规则】\n"
                    "- 加班后可申请调休，调休时长=加班时长\n"
                    "- 工作日加班：1:1调休\n"
                    "- 休息日加班：1:1.5调休（加班8小时=调休12小时）\n"
                    "- 法定节假日加班：1:2调休或支付3倍工资\n"
                    "- 调休需提前1天申请，3个月内有郊\n"
                    "- 过期未休的调休自动清零"
                ),
                "category": "请假-调休",
                "keywords": ["调休", "补休", "加班", "调休时长", "加班工资"]
            },
            # -- 丧假 --
            {
                "content": (
                    "【丧假（哀悼假）规则】\n"
                    "- 直系亲属（父母/配偶/子女）：3天\n"
                    "- 祖父母/外祖父母/兄弟姐妹：1天\n"
                    "- 其他亲属：无带薪丧假，可申请事假\n"
                    "- 丧假为带薪假，需提供相关证明\n"
                    "- 可根据路程远近酌情增加路途假（不超过2天）"
                ),
                "category": "请假-丧假",
                "keywords": ["丧假", "奔丧", "亲属", "去世", "哀悼"]
            },
            # -- 请假通用规则 --
            {
                "content": (
                    "【请假通用规则与FAQ】\n"
                    "一、审批链\n"
                    "- 1天：直属主管\n"
                    "- 2-3天：主管→部门经理\n"
                    "- 3天以上：主管→部门经理→部门总监\n"
                    "- 特殊假期（婚/产/丧）：需额外HR审批\n"
                    "二、请假冲突\n"
                    "- 同一团队同一天请假人数不得超过团队总人数的30%\n"
                    "- 项目关键节点前一周原则上不批长假(≥3天)\n"
                    "三、FAQ\n"
                    "Q: 请假申请提交后多久审批？\n"
                    "A: 一般1个工作日内审批，紧急可电话/飞书催办。\n"
                    "Q: 请假被拒怎么办？\n"
                    "A: 可向上级主管申诉或联系HR协调。\n"
                    "Q: 半天假怎么算？\n"
                    "A: 上午半天=9:00-13:00，下午半天=14:00-18:00。\n"
                    "Q: 年假没用完可以换钱吗？\n"
                    "A: 正常在职年假不折现，离职时统一结算。"
                ),
                "category": "请假-通用规则",
                "keywords": ["请假", "审批链", "冲突", "半天", "FAQ", "年假折现", "催办", "被拒"]
            },

            # ================================================================
            # 四、报销制度
            # ================================================================
            {
                "content": (
                    "差旅报销标准：国内出差住宿费一线城市（北上广深）≤500元/晚，"
                    "其他城市≤350元/晚。交通费高铁二等座/飞机经济舱实报实销。"
                    "市内交通≤100元/天。餐饮补贴80元/天（无需发票）。"
                    "出差需提前提交出差申请单，返回后5个工作日内提交报销。"
                ),
                "category": "报销-差旅",
                "keywords": ["差旅", "出差", "住宿", "交通费", "餐饮补贴", "报销标准"]
            },
            {
                "content": (
                    "报销流程：1. 登录OA系统→费用报销→新建报销单 "
                    "2. 选择报销类型（差旅/办公/餐费/交通） "
                    "3. 填写报销金额和事由 4. 上传发票照片"
                    "（金额≥2000元需上传原始发票） "
                    "5. 提交→直属主管审批→财务审核 "
                    "6. 审批通过后5个工作日内打款到工资卡。"
                    "发票必须真实有效，电子发票和纸质发票具有同等效力。"
                ),
                "category": "报销-流程",
                "keywords": ["报销流程", "OA系统", "发票", "审批", "打款", "电子发票"]
            },
            {
                "content": (
                    "办公用品采购与报销：单次采购金额≤500元可自行购买后报销，"
                    ">500元需提前申请采购。报销时需上传购物小票或发票照片。"
                    "IT类设备（键盘/鼠标/显示器等）需通过IT部门统一采购，不得自行购买报销。"
                ),
                "category": "报销-办公用品",
                "keywords": ["办公用品", "采购", "小票", "设备", "自行购买", "IT设备"]
            },

            # ================================================================
            # 五、行政服务
            # ================================================================
            {
                "content": (
                    "会议室预定规则：小型会议室（4-6人）可随时预定，"
                    "中大型会议室（8-20人）需提前1天预定。"
                    "每周一上午为部门周例会预留时间，不可预定。"
                    "会议室配备投影仪和白板，部分会议室支持视频会议。"
                    "预定后15分钟未签到使用，系统自动释放。"
                ),
                "category": "行政-会议室",
                "keywords": ["会议室", "预定", "投影仪", "白板", "视频会议", "签到"]
            },
            {
                "content": (
                    "快递寄送服务：公司提供顺丰月结账号用于公务快递。"
                    "个人快递不得使用公司账号。"
                    "寄件流程：填写快递单→交至前台→前台统一发出。"
                    "每日快递取件时间：上午10:00和下午16:00。"
                    "紧急文件可使用同城闪送（需主管审批）。"
                ),
                "category": "行政-快递",
                "keywords": ["快递", "顺丰", "寄件", "月结", "前台", "闪送", "取件"]
            },
            {
                "content": (
                    "访客登记流程：访客需提前由接待员工在OA→行政服务→访客登记中"
                    "提交访客信息（姓名/手机号/身份证号/到访时间/接待人）。"
                    "审批通过后，访客将收到短信通知及电子访客码。"
                    "访客持码在前台扫码通行。访客码当日有效，过期自动失效。"
                ),
                "category": "行政-访客",
                "keywords": ["访客", "登记", "接待", "电子码", "前台", "通行", "预约"]
            },
            {
                "content": (
                    "资产领用与归还：公司资产（电脑/显示器/电话/打印机等）"
                    "通过OA→行政服务→资产领用申请。"
                    "新员工入职当天由IT部门统一发放标配设备。"
                    "离职员工需在最后一个工作日归还所有公司资产，"
                    "由行政部门确认签字后方可办理离职手续。"
                ),
                "category": "行政-资产管理",
                "keywords": ["资产", "领用", "电脑", "设备", "离职", "归还", "新员工"]
            },

            # ================================================================
            # 六、新员工入职与设备配备 (v7 新增)
            # ================================================================
            {
                "content": (
                    "【新员工入职IT设备配备标准】\n"
                    "各岗位标配如下——\n\n"
                    "一、前端开发工程师\n"
                    "- 笔记本: MacBook Pro 16\" M4 Pro 或 ThinkPad X1 Carbon Gen 12\n"
                    "- 显示器: Dell U2723QE 4K 27\" 或 LG 27UP850N 4K 27\"\n"
                    "- 键鼠: 罗技 MX Keys + MX Master 3S\n"
                    "- 耳机: Sony WH-1000XM5 降噪耳机\n"
                    "- 备用: USB-C Hub、4K HDMI线\n\n"
                    "二、后端开发工程师\n"
                    "- 笔记本: ThinkPad X1 Carbon Gen 12 或 MacBook Pro 14\"\n"
                    "- 显示器: Dell U2723QE 4K 27\" ×2（双屏）\n"
                    "- 键鼠: 机械键盘(Filco/Leopold) + 罗技 MX Master 3S\n"
                    "- 耳机: Sony WH-1000XM5\n\n"
                    "三、设计师\n"
                    "- 笔记本: MacBook Pro 16\" M4 Max + 64GB\n"
                    "- 显示器: Apple Studio Display 27\" 5K 或 Dell U3224KB 6K\n"
                    "- 数位板: Wacom Intuos Pro M\n"
                    "- 键鼠: Apple Magic Keyboard + Magic Trackpad\n\n"
                    "四、产品经理/运营/HR\n"
                    "- 笔记本: ThinkPad X1 Carbon Gen 12 或 MacBook Air 15\"\n"
                    "- 显示器: Dell U2723QE 4K 27\" ×1\n"
                    "- 键鼠: 罗技 MX Keys Mini + MX Anywhere 3S\n\n"
                    "五、通用规则\n"
                    "- 新员工入职前1周，HR将入职通知同步IT部门\n"
                    "- IT在入职前3个工作日完成设备准备\n"
                    "- 特殊设备需求（如高性能GPU）需额外申请审批"
                ),
                "category": "行政-新员工入职",
                "keywords": ["入职", "新员工", "设备", "标配", "前端", "后端", "笔记本", "显示器",
                           "键鼠", "耳机", "MacBook", "ThinkPad", "Dell", "罗技", "电脑配置"]
            },
            {
                "content": (
                    "【IT设备领用流程】\n"
                    "一、适用对象\n"
                    "- 新入职员工（入职当天领取标配设备）\n"
                    "- 在职员工（设备更换/升级/借用）\n\n"
                    "二、新员工领用流程\n"
                    "1. HR在入职前1周提交「新员工设备申请」工单\n"
                    "2. IT部门根据岗位标配准备设备\n"
                    "3. 入职当天上午10:00，新员工到IT服务台(305室)签领\n"
                    "4. 签领时核验设备清单并签字确认\n"
                    "5. 设备信息录入资产管理系统\n\n"
                    "三、在职员工领用流程\n"
                    "1. 员工提交 admin 类型工单（service_type=资产领用）\n"
                    "2. 直属主管审批（1个工作日内）\n"
                    "3. IT部门确认库存并配发\n"
                    "4. 更换设备需归还旧设备后方可领取新设备\n\n"
                    "四、设备归还\n"
                    "- 离职员工须在最后一个工作日归还所有IT设备\n"
                    "- 设备经IT检测确认无损坏后签字\n"
                    "- 人为损坏需照价赔偿（从最后薪资扣除）\n"
                    "- 笔记本电脑3年更换周期/显示器5年/外设2年"
                ),
                "category": "行政-新员工入职",
                "keywords": ["领用", "流程", "入职", "签领", "IT服务台", "305", "资产",
                           "归还", "更换", "离职", "审批"]
            },
            {
                "content": (
                    "【办公物品采购流程与审批】\n"
                    "一、采购金额审批链\n"
                    "- 单次采购金额 <500元：直属主管审批即可\n"
                    "- 500元 ≤ 金额 <5000元：主管→部门经理审批\n"
                    "- 金额 ≥5000元：主管→部门经理→总监审批，且需三家比价\n"
                    "- 金额 ≥50000元：需CEO审批+采购委员会评审\n\n"
                    "二、IT设备采购特别规定\n"
                    "- 所有IT类设备（电脑/显示器/外设/服务器等）统一由IT部门采购\n"
                    "- 员工不得自行购买后报销IT设备\n"
                    "- IT设备须从公司认证供应商列表中选择\n"
                    "- 当前认证供应商：戴尔(Dell)、联想(Lenovo)、苹果(Apple)、罗技(Logitech)\n\n"
                    "三、采购流程\n"
                    "1. 需求方提交 admin 类型工单（service_type=采购申请）\n"
                    "2. 注明物品名称、规格型号、数量、预算、紧急程度\n"
                    "3. 按金额走对应审批链\n"
                    "4. 审批通过后IT/行政下单采购\n"
                    "5. 到货后验收→入库→通知领用人\n"
                    "6. 整个流程预计3-10个工作日（取决于审批速度和供应商库存）\n\n"
                    "四、紧急采购\n"
                    "- P0/P1紧急需求可走绿色通道，事后补审批\n"
                    "- 绿色通道需部门总监+IT总监双签"
                ),
                "category": "行政-新员工入职",
                "keywords": ["采购", "审批", "金额", "供应商", "比价", "绿色通道", "紧急",
                           "Dell", "联想", "苹果", "罗技", "预算", "入库"]
            },
            {
                "content": (
                    "【常见问题FAQ：设备与采购】\n"
                    "Q1: 新员工入职多久能拿到电脑？\n"
                    "A: 入职当天上午即可领取。HR提前1周通知IT，库存充足的情况下设备会提前准备。\n\n"
                    "Q2: 标配设备不满足需求怎么办？\n"
                    "A: 提交「资产领用」工单说明特殊需求，主管审批后IT评估是否可升级配置。\n\n"
                    "Q3: 设备损坏了怎么处理？\n"
                    "A: 非人为损坏免费维修/更换。人为损坏需照价赔偿。联系IT服务台(分机8888)报修。\n\n"
                    "Q4: 可以自己买设备然后报销吗？\n"
                    "A: IT类设备（电脑/显示器/外设）不可以。必须通过IT部门统一采购。\n"
                    "    非IT类办公用品（如笔、本子）<500元可自行购买后走报销流程。\n\n"
                    "Q5: 采购审批一般多久？\n"
                    "A: <500元: 1个工作日内。500-5000元: 2-3个工作日。≥5000元: 5-10个工作日（含比价）。\n\n"
                    "Q6: 库存显示缺货怎么办？\n"
                    "A: 自动触发采购流程。IT会根据最低库存阈值自动补货。紧急需求可走绿色通道。"
                ),
                "category": "行政-新员工入职",
                "keywords": ["FAQ", "常见问题", "入职", "设备", "采购", "损坏", "维修",
                           "报销", "审批时间", "缺货", "补货"]
            },
        ]

    async def initialize(self):
        """初始化知识库服务"""
        try:
            existing_docs = self.db.get_all_documents()

            if not existing_docs:
                logger.info("数据库为空，初始化默认知识库")
                await self._create_default_knowledge()
            else:
                logger.info(f"从数据库加载了 {len(existing_docs)} 条知识")

            await self._build_vector_index()
            self.initialized = True
            logger.info("知识库服务初始化完成")

        except Exception as e:
            logger.error(f"知识库服务初始化失败: {e}")
            raise

    async def _create_default_knowledge(self):
        """创建默认知识库"""
        for knowledge in self.default_knowledge:
            try:
                text_for_embedding = f"{knowledge['content']} {' '.join(knowledge['keywords'])}"
                embedding = embed_input(text_for_embedding)

                self.db.add_document(
                    content=knowledge['content'],
                    category=knowledge['category'],
                    keywords=knowledge['keywords'],
                    embedding=embedding
                )
                logger.debug(f"添加默认知识: {knowledge['content'][:50]}...")

            except Exception as e:
                logger.error(f"添加默认知识失败: {e}")

    # ============================================================
    # FAISS 向量索引 — v2 IndexIDMap
    # ============================================================

    async def _build_vector_index(self):
        """
        构建向量索引 — 使用 IndexIDMap 支持真删除

        IndexIDMap 包装 IndexFlatIP，使 remove_ids() 可用。
        """
        try:
            documents = self.db.get_all_documents()
            if not documents:
                logger.warning("没有文档可用于构建索引")
                return

            embeddings = []
            doc_ids = []

            for doc in documents:
                if doc.get('embedding'):
                    embeddings.append(doc['embedding'])
                    doc_ids.append(doc['id'])
                else:
                    logger.warning(f"文档 {doc['id']} 缺少嵌入向量，正在生成...")
                    text_for_embedding = (
                        f"{doc['content']} {' '.join(doc.get('keywords', []))}"
                    )
                    embedding = embed_input(text_for_embedding)
                    self.db.update_document(doc['id'], embedding=embedding)
                    embeddings.append(embedding)
                    doc_ids.append(doc['id'])

            if embeddings:
                embeddings_array = np.array(embeddings).astype('float32')
                dimension = embeddings_array.shape[1]

                # v2: IndexIDMap 包装，支持 remove_ids
                base_index = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIDMap(base_index)
                self.index.add_with_ids(
                    embeddings_array,
                    np.array(doc_ids, dtype=np.int64),
                )
                logger.info(
                    f"构建向量索引完成 (IndexIDMap)，包含 {len(embeddings)} 个向量，"
                    f"维度={dimension}"
                )
            else:
                logger.warning("没有有效的嵌入向量，无法构建索引")

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            raise

    async def _add_to_index(self, doc_id: int, embedding: list):
        """
        增量添加单个向量到 IndexIDMap

        IndexIDMap 支持 add_with_ids，每个向量绑定 doc_id 作为外部 ID。
        """
        if self.index is None:
            await self._build_vector_index()
            return

        emb_array = np.array([embedding]).astype('float32')
        id_array = np.array([doc_id], dtype=np.int64)
        self.index.add_with_ids(emb_array, id_array)
        logger.debug(f"增量添加向量: doc_id={doc_id}, index_size={self.index.ntotal}")

    async def _remove_from_index(self, doc_id: int):
        """从索引中删除指定 doc_id 的向量（IndexIDMap 真删除）"""
        if self.index is None:
            return

        try:
            id_array = np.array([doc_id], dtype=np.int64)
            n_before = self.index.ntotal
            self.index.remove_ids(id_array)
            n_removed = n_before - self.index.ntotal
            if n_removed > 0:
                logger.debug(f"从索引删除向量: doc_id={doc_id}, removed={n_removed}")
        except Exception as e:
            logger.warning(f"从索引删除向量失败 (doc_id={doc_id}): {e}")

    # ============================================================
    # 搜索
    # ============================================================

    async def search(self, query: str, top_k: int = 3, category: str = None) -> List[Dict]:
        """
        向量搜索相关文档

        Args:
            query: 查询文本
            top_k: 返回文档数
            category: 可选分类过滤

        Returns:
            相关文档列表（含 score / rank）
        """
        if not self.initialized or self.index is None:
            logger.warning("知识库服务未初始化或索引不可用")
            return []

        try:
            query_embedding = embed_input(query)
            query_array = np.array([query_embedding]).astype('float32')

            # 多检索一些候选
            k = min(top_k * 2, self.index.ntotal)
            if k == 0:
                return []

            scores, indices = self.index.search(query_array, k)

            # IndexIDMap 的 search 返回：indices 是外部 doc_id
            # scores 是内积相似度（IndexFlatIP 内积越高越相似）
            results = []
            for score, doc_id in zip(scores[0], indices[0]):
                doc_id = int(doc_id)
                if doc_id < 0:  # FAISS 无效索引标记
                    continue

                doc = self.db.get_document(doc_id)
                if not doc:
                    continue

                # 分类过滤
                if category and doc.get('category') != category:
                    continue

                doc['score'] = float(score)
                doc['rank'] = len(results) + 1
                results.append(doc)

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.error(f"搜索知识库失败: {e}")
            return []

    # ============================================================
    # CRUD
    # ============================================================

    async def add_document(self, content: str, category: str, keywords: List[str] = None) -> bool:
        """添加新文档（增量更新索引）"""
        try:
            if keywords is None:
                keywords = []

            text_for_embedding = f"{content} {' '.join(keywords)}"
            embedding = embed_input(text_for_embedding)

            doc_id = self.db.add_document(content, category, keywords, embedding)
            await self._add_to_index(doc_id, embedding)

            logger.info(f"成功添加文档 {doc_id}: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    async def update_document(
        self, doc_id: int, content: str = None, category: str = None,
        keywords: List[str] = None,
    ) -> bool:
        """更新文档 — 删除旧向量 + 添加新向量（真替换）"""
        try:
            embedding = None
            if content is not None or keywords is not None:
                current_doc = self.db.get_document(doc_id)
                if not current_doc:
                    return False

                final_content = content if content is not None else current_doc['content']
                final_keywords = keywords if keywords is not None else current_doc.get('keywords', [])

                text_for_embedding = f"{final_content} {' '.join(final_keywords)}"
                embedding = embed_input(text_for_embedding)

            success = self.db.update_document(doc_id, content, category, keywords, embedding)

            if success and embedding is not None:
                # 真替换：先删除旧向量，再添加新向量
                await self._remove_from_index(doc_id)
                await self._add_to_index(doc_id, embedding)

            return success

        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False

    async def delete_document(self, doc_id: int, soft_delete: bool = True) -> bool:
        """
        删除文档 — IndexIDMap 支持真删除

        不再依赖脏数据累积+全量重建。
        """
        try:
            success = self.db.delete_document(doc_id, soft_delete)

            if success:
                await self._remove_from_index(doc_id)

            return success

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    # ============================================================
    # 查询
    # ============================================================

    def get_all_documents(self, include_inactive: bool = False) -> List[Dict]:
        """获取所有文档"""
        return self.db.get_all_documents(include_inactive)

    def get_document(self, doc_id: int) -> Dict:
        """获取指定文档"""
        return self.db.get_document(doc_id)

    def get_all_categories(self) -> List[str]:
        """获取所有分类"""
        return self.db.get_all_categories()

    def get_documents_count(self) -> int:
        """获取文档总数"""
        return self.db.get_documents_count()

    def search_by_category(self, category: str) -> List[Dict]:
        """按分类搜索文档"""
        return self.db.search_documents_by_category(category)

    def search_by_keywords(self, keywords: List[str]) -> List[Dict]:
        """按关键词搜索文档"""
        return self.db.search_documents_by_keywords(keywords)

    # ============================================================
    # 用户反馈
    # ============================================================

    def record_feedback(
        self, doc_id: int, is_helpful: bool, user_id: str = "", comment: str = "",
    ) -> bool:
        """
        记录用户对文档的反馈（点赞/踩）

        Args:
            doc_id: 文档ID
            is_helpful: True=有帮助, False=无帮助
            user_id: 用户标识
            comment: 补充说明
        """
        try:
            from db.models import DocumentFeedback

            session_manager = self.db_router.session_manager
            with session_manager.session_scope() as session:
                feedback = DocumentFeedback(
                    doc_id=doc_id,
                    is_helpful=1 if is_helpful else 0,
                    user_id=user_id,
                    comment=comment,
                )
                session.add(feedback)

            logger.info(
                f"反馈已记录: doc={doc_id}, helpful={is_helpful}, user={user_id}"
            )
            return True

        except Exception as e:
            logger.error(f"记录反馈失败: {e}")
            return False

    def get_feedback_stats(self, doc_id: int) -> dict:
        """
        获取文档反馈统计

        Returns:
            {total, helpful_count, not_helpful_count, helpful_ratio}
        """
        try:
            from db.models import DocumentFeedback
            from sqlalchemy import func

            session_manager = self.db_router.session_manager
            with session_manager.session_scope() as session:
                total = session.query(DocumentFeedback).filter(
                    DocumentFeedback.doc_id == doc_id,
                ).count()

                helpful = session.query(DocumentFeedback).filter(
                    DocumentFeedback.doc_id == doc_id,
                    DocumentFeedback.is_helpful == 1,
                ).count()

                return {
                    "doc_id": doc_id,
                    "total": total,
                    "helpful_count": helpful,
                    "not_helpful_count": total - helpful,
                    "helpful_ratio": round(helpful / max(total, 1), 2),
                }

        except Exception as e:
            logger.error(f"获取反馈统计失败: {e}")
            return {"doc_id": doc_id, "total": 0, "helpful_count": 0,
                    "not_helpful_count": 0, "helpful_ratio": 0}
