import streamlit as st
import plotly.express as px

import os
import sys

# robust import: prefer package import; ensure project root is on sys.path when running via streamlit
try:
    from webapp.utils import load_summary, list_candidates, load_equity, load_trades, list_runs, load_meta
    from webapp.utils import load_fold_metrics, load_fold_ranges
except ImportError:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from webapp.utils import load_summary, list_candidates, load_equity, load_trades, list_runs, load_meta
    from webapp.utils import load_fold_metrics, load_fold_ranges

import pandas as pd
# 混淆矩阵（若后续提供逐样本标签/预测，可启用）
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title='ForecastLab Prediction Review', layout='wide')

st.sidebar.title('ForecastLab Review')

# --- Query-param persistence helpers ---
# Streamlit 1.30+ exposes st.query_params (preferred) but older versions use experimental_get/set_query_params.
try:
    _QP = st.query_params  # type: ignore[attr-defined]

    def _qp_get(key: str, default: str = '') -> str:
        v = _QP.get(key, default)
        if isinstance(v, list):
            return v[0] if v else default
        return v

    def _qp_get_list(key: str) -> list:
        v = _QP.get(key, [])
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    def _qp_set(key: str, value):
        _QP[key] = value

except Exception:  # pragma: no cover

    def _qp_get(key: str, default: str = '') -> str:
        v = st.experimental_get_query_params().get(key, [default])
        return v[0] if v else default

    def _qp_get_list(key: str) -> list:
        return st.experimental_get_query_params().get(key, [])

    def _qp_set(key: str, value):
        st.experimental_set_query_params(**{key: value})


def _safe_int(s: str, default: int) -> int:
    try:
        return int(s)
    except Exception:
        return default


# Select backtest run directory
runs_root = os.path.join('outputs', 'backtest')
run_names = list_runs(runs_root) if 'list_runs' in globals() else []

# restore run from query param if possible
_qp_run = _qp_get('run', '')
run_default = run_names[0] if run_names else None
run_index = run_names.index(_qp_run) if (_qp_run in run_names) else 0

run_sel = st.sidebar.selectbox('选择回测run（结果目录）', options=run_names, index=run_index, key='run_sel') if run_names else None
if run_sel:
    _qp_set('run', run_sel)

base_dir = os.path.join(runs_root, run_sel) if run_sel else runs_root

summary = load_summary(base_dir)
meta = load_meta(base_dir)

# Sidebar quick meta
with st.sidebar.expander('Run 元信息', expanded=True):
    if meta:
        st.write({
            'data_source': meta.get('data_source'),
            'data_path': meta.get('data_path'),
            'feature_set': meta.get('feature_set'),
            'stage': meta.get('stage'),
            'init_train': meta.get('init_train'),
            'oos_window': meta.get('oos_window'),
            'step': meta.get('step'),
            'model_preference': meta.get('model_preference'),
            'lgb_installed': meta.get('lgb_installed'),
        })
    else:
        st.caption('该 run 缺少 meta.json（建议重新跑脚本生成）')

# 顶部 Tabs（移除“策略与交易”，聚焦预测评估）
tabs = st.tabs([
    '总览', '数据与标签', '训练与预测', '候选对比', '文档'
])

# 总览（仅基于分类评估指标）
with tabs[0]:
    st.header('回测结果总览（预测评估）')
    if summary.empty:
        st.warning('找不到 summary CSV：outputs/backtest/reversal_param_search_wf_results.csv')
    else:
        # 排序与筛选（加入准确率与平衡准确率）
        sort_candidates = ['f1_median', 'accuracy_median', 'balanced_accuracy_median', 'folds', 'T', 'X']
        sort_col = st.selectbox('排序列', [c for c in sort_candidates if c in summary.columns])
        st.caption('字段说明：\n- f1_median：各折 Macro F1 的中位数，关注三类总体识别质量。\n- accuracy_median：各折整体准确率的中位数。\n- balanced_accuracy_median：各折平衡准确率（各类召回均值）的中位数。\n- folds：walk-forward OOS 折数。\n- T/X：标签参数（未来窗口/阈值）。')
        with st.expander('candidate 命名示例（点击展开）', expanded=False):
            st.markdown('- 例如 candidate = `T7_X6`：\n  - T=7：观察/标签窗口 7 天。\n  - X=0.06：阈值 6%，未来 7 天内若上涨/下跌幅度 ≥ 6% 分别打 +1/-1 标签，否则 0。\n- 例如 candidate = `T21_X10`：T=21 天，X=0.10（10%）。')
        ascending = st.checkbox('升序', value=False)
        df_view = summary.sort_values([sort_col], ascending=ascending)
        # 展示与预测评估相关的核心列
        base_cols = ['candidate','T','X','folds','f1_median','precision_median','recall_median','accuracy_median','balanced_accuracy_median']
        cols_to_show = [c for c in base_cols if c in df_view.columns]
        st.dataframe(df_view[cols_to_show], width='stretch')
        # TopN
        cols = st.columns(3)
        with cols[0]:
            top_n = st.number_input('Top N', min_value=1, max_value=50, value=5)
        st.dataframe(df_view[cols_to_show].head(top_n), width='stretch')

# 数据与标签（展示基本数据文件信息与标签定义）
with tabs[1]:
    st.header('数据与标签')

    # Display meta if available
    if meta:
        st.markdown(f"- 数据源（run）：`{meta.get('data_source', 'unknown')}`")
        st.markdown(f"- 数据路径（run）：`{meta.get('data_path', '')}`")
        st.markdown(f"- feature_set（run）：`{meta.get('feature_set', '')}`")
        if meta.get('selected_features'):
            st.markdown(f"- 特征列（run）：{', '.join(meta.get('selected_features'))}")
    else:
        st.markdown('- 数据源：未提供 meta.json（展示为默认演示）')

    st.markdown('- 标签：基于未来 T 天的最大/最小价计算 upswing/drawdown，阈值 X 判定，输出 {-1,0,1}')
    # 展示 summary 中不同 (T,X) 的分布（按 F1 上色）
    if not summary.empty and all(k in summary.columns for k in ['T','X','f1_median']):
        st.subheader('候选参数分布（T, X）')
        fig_scatter = px.scatter(summary, x='T', y='X', size=None, color='f1_median', title='候选参数分布（按 F1 着色）')
        st.plotly_chart(fig_scatter, width='stretch')
    st.caption('评估指标说明：\n- Macro F1（f1_median）：各类F1简单平均的稳健聚合，用于排序。\n- Accuracy：整体正确率，类不平衡时可能偏高。\n- Balanced Accuracy：对每类召回率的平均，抵抗不平衡。\n- Precision/Recall（macro）：分别衡量预测的准确性与检出能力。')

    st.subheader('样本构造可视化（标签生成演示）')
    # 选择 T/X 并演示标签生成
    col_cfg = st.columns(3)
    with col_cfg[0]:
        T_demo = st.number_input('T（未来窗口，天）', min_value=3, max_value=60, value=int(summary['T'].median()) if 'T' in summary.columns else 14)
    with col_cfg[1]:
        X_demo = st.number_input('X（阈值，比例）', min_value=0.01, max_value=0.3, step=0.01, value=float(summary['X'].median()) if 'X' in summary.columns else 0.08)
    with col_cfg[2]:
        show_n = st.number_input('展示最近 N 天', min_value=50, max_value=500, value=200)

    st.subheader('walk-forward 时间范围（按折计算）')
    st.caption('OOS = Out-of-Sample（样本外/场外测试）。业务含义：用历史数据训练模型（训练集），再在未来从未参与训练的时间区间上评估/预测（OOS测试集），用于模拟真实上线后的表现，避免“用未来信息作弊”。')

    with st.expander('fold（折）如何对应到“起始日期/结束日期”？（点击展开）', expanded=False):
        st.markdown(
            """
- **fold = k** 表示第 k 次 *walk-forward* 时间切分（从 0 开始，按时间顺序向前滚动）。每一折都会得到两段连续区间：
  - **训练集**：从全数据的**起始日期**开始，到某个“训练结束日期”为止；
  - **OOS 测试集**：紧接训练集之后的一段“未来区间”，从“OOS 起始日期”到“OOS 结束日期”。

- 设：
  - `init_train` = 初始训练集长度（行/天数）
  - `step` = 每折向前滚动的步长（行/天数）
  - `oos_window` = 每折 OOS 测试窗口长度（行/天数）

- 则第 **k 折**的切分点（用行索引表示）是：
  - `train_end_ix = init_train + k * step`
  - 训练集覆盖：`[0, train_end_ix-1]`
  - OOS 覆盖：`[train_end_ix, train_end_ix + oos_window - 1]`

- 将上面索引映射到日期（你的数据按时间升序）：
  - **训练集开始日期** = 数据第 0 行日期（全数据起始日期）
  - **训练集结束日期** = 第 `train_end_ix-1` 行日期
  - **OOS开始日期** = 第 `train_end_ix` 行日期
  - **OOS结束日期** = 第 `train_end_ix+oos_window-1` 行日期

- 直观例子：
  - **fold=0**：训练集
  - **fold=1**：训练结束点在 fold=0 的基础上**再往后推进 `step` 条**，因此训练集更长、OOS 整体也更靠后。

> 注：这里的“天”本质是数据行数（日频则≈自然日，但仍以交易序列为准）。
            """
        )

    col_wf = st.columns(5)
    with col_wf[0]:
        init_train_demo = st.number_input('init_train（天）', min_value=50, max_value=5000, value=500)
    with col_wf[1]:
        step_demo = st.number_input('step（天）', min_value=1, max_value=365, value=21)
    with col_wf[2]:
        oos_window_demo = st.number_input('oos_window（天）', min_value=1, max_value=365, value=63)
    with col_wf[3]:
        fold_demo = st.number_input('fold（从0开始）', min_value=0, max_value=500, value=0)
    with col_wf[4]:
        show_folds_demo = st.number_input('展示前 N 折', min_value=1, max_value=200, value=10)

    show_mode = st.radio('展示模式', ['单折', '多折(前N折)'], horizontal=True)

    # 读取数据并计算基础特征与标签
    try:
        # In-demo data loading: prefer run's data_path, fallback to Yahoo
        demo_data_path = meta.get('data_path') if meta.get('data_path') else os.path.join('data', 'raw', 'btc_yahoo_data.csv')
        df_raw = pd.read_csv(demo_data_path)
        # 规范列名（兼容 Yahoo / Binance）
        if 'Date' in df_raw.columns:
            df_raw.rename(columns={'Date': 'timestamp', 'Close': 'close', 'Open':'open','High':'high','Low':'low','Volume':'volume'}, inplace=True)
        df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
        df = df_raw.copy()
        df['close'] = df['close'].astype(float)

        # 计算并展示 walk-forward 时间范围（基于整段数据 df_raw）
        # 注：这里展示的是“时间切分”，与标签(T/X)演示是两个独立维度。
        n_total = len(df_raw)
        init_n = int(init_train_demo)
        step_n = int(step_demo)
        oos_n = int(oos_window_demo)

        def build_wf_row(fold_n: int):
            train_end_ix = init_n + fold_n * step_n
            test_start_ix = train_end_ix
            test_end_ix = train_end_ix + oos_n
            if train_end_ix <= 0 or test_end_ix > n_total:
                return {
                    'fold': fold_n,
                    '训练集开始日期': None,
                    '训练集结束日期': None,
                    'OOS开始日期': None,
                    'OOS结束日期': None,
                    '备注': f'索引超界：train_end_ix={train_end_ix}, test_end_ix={test_end_ix}, n_total={n_total}'
                }
            return {
                'fold': fold_n,
                '训练集开始日期': df_raw['timestamp'].iloc[0],
                '训练集结束日期': df_raw['timestamp'].iloc[train_end_ix - 1],
                'OOS开始日期': df_raw['timestamp'].iloc[test_start_ix],
                'OOS结束日期': df_raw['timestamp'].iloc[test_end_ix - 1],
                '备注': ''
            }

        if show_mode == '单折':
            fold_n = int(fold_demo)
            wf_table = pd.DataFrame([build_wf_row(fold_n)])
        else:
            # 多折：展示前 N 折（自动截断到最大可用折数）
            max_folds_possible = max(0, (n_total - init_n - oos_n) // step_n + 1)
            n_show = min(int(show_folds_demo), max_folds_possible)
            wf_table = pd.DataFrame([build_wf_row(f) for f in range(n_show)])

        st.dataframe(wf_table, width='stretch')

        # 计算未来窗口的最大/最小价并生成 upswing/drawdown
        T_int = int(T_demo)
        future_max = df['close'].shift(-1).rolling(T_int).max()
        future_min = df['close'].shift(-1).rolling(T_int).min()
        df['future_upswing'] = (future_max - df['close']) / df['close']
        df['future_drawdown'] = (df['close'] - future_min) / df['close']
        df['label_top'] = (df['future_drawdown'] >= X_demo).astype(int)
        df['label_bottom'] = (df['future_upswing'] >= X_demo).astype(int)
        df['label'] = 0
        df.loc[df['label_top'] == 1, 'label'] = -1
        df.loc[df['label_bottom'] == 1, 'label'] = 1
        df = df.dropna().reset_index(drop=True)

        # 最近窗口展示
        df_show = df.tail(show_n).copy()
        st.markdown('标签规则：若未来窗口内最大涨幅≥X则标记为 +1；若最大跌幅≥X则标记为 -1；否则 0。')

        # 样本时间范围可视化（表格展示）
        st.subheader('样本时间范围')
        # 该区块对应“演示用”的 (T_demo, X_demo)，并非回测summary里的候选；这里生成一个candidate展示名便于对齐。
        candidate_demo = f"T{T_int}_X{int(round(float(X_demo) * 100))}"
        info_table = pd.DataFrame({
            'candidate(演示参数)': [candidate_demo],
            'T(天)': [T_int],
            'X(阈值)': [float(X_demo)],
            '样本总数': [len(df_show)],
            '开始日期': [df_show['timestamp'].iloc[0]],
            '结束日期': [df_show['timestamp'].iloc[-1]]
        })
        st.dataframe(info_table, width='stretch')

        # 每条样本的起始日期（即当前行的timestamp）+ 标签 + 未来窗口覆盖区间
        sample_table = df_show[['timestamp', 'label']].copy()
        sample_table.rename(columns={'timestamp': '样本起始日期', 'label': '标签'}, inplace=True)

        # 未来窗口：按标签构造逻辑，窗口从 t+1 开始，长度为 T
        # 注：基于数据行序而非自然日（交易日序列）。
        sample_table['未来窗口开始日期'] = df_show['timestamp'].shift(-1).values
        sample_table['未来窗口结束日期'] = df_show['timestamp'].shift(-T_int).values

        st.dataframe(sample_table.reset_index(drop=True), width='stretch')

        # 价格 + 信号（带滑动时间轴）
        st.subheader('价格与信号（可滑动）')
        fig_price = px.line(df_show, x='timestamp', y='close', title=f'收盘价与信号（最近{show_n}天）')
        # 标注信号点
        df_sig_up = df_show[df_show['label'] == 1]
        df_sig_dn = df_show[df_show['label'] == -1]
        fig_price.add_scatter(x=df_sig_up['timestamp'], y=df_sig_up['close'], mode='markers', name='Upswing(+1)', marker=dict(color='green', symbol='triangle-up', size=8))
        fig_price.add_scatter(x=df_sig_dn['timestamp'], y=df_sig_dn['close'], mode='markers', name='Drawdown(-1)', marker=dict(color='red', symbol='triangle-down', size=8))
        fig_price.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='date'))
        st.plotly_chart(fig_price, width='stretch')

        # 标签分布柱状图
        st.subheader('标签分布（柱状图）')
        vc_show = df_show['label'].value_counts().sort_index()
        fig_bar = px.bar(x=vc_show.index.astype(str), y=vc_show.values, labels={'x':'标签','y':'样本数'}, title='最近窗口标签分布')
        st.plotly_chart(fig_bar, width='stretch')
        st.write({'class_counts_recent': {'-1': int(vc_show.get(-1,0)), '0': int(vc_show.get(0,0)), '1': int(vc_show.get(1,0))}})

        # 未来涨跌幅与阈值线（可滑动）
        st.subheader('未来窗口涨跌幅与阈值（可滑动）')
        fig_ud = px.line(df_show, x='timestamp', y=['future_upswing','future_drawdown'], title='未来窗口涨跌幅')
        fig_ud.add_hline(y=X_demo, line_dash='dot', line_color='green', annotation_text=f'阈值 +{X_demo:.2f}')
        fig_ud.add_hline(y=X_demo, line_dash='dot', line_color='red', annotation_text=f'阈值 -{X_demo:.2f}')
        fig_ud.update_layout(xaxis=dict(rangeslider=dict(visible=True), type='date'))
        st.plotly_chart(fig_ud, width='stretch')

        # 类分布与示例片段
        vc = df['label'].value_counts().sort_index()
        st.write({'class_counts_total': {'-1': int(vc.get(-1,0)), '0': int(vc.get(0,0)), '1': int(vc.get(1,0))}})
        st.dataframe(df_show[['timestamp','close','future_upswing','future_drawdown','label']].head(50), width='stretch')
    except Exception as e:
        st.warning(f'样本构造演示失败：{e}')

# 训练与预测（展示分类指标 + 候选每折明细）
with tabs[2]:
    st.header('训练与预测')
    if summary.empty:
        st.info('暂无 summary 数据')
    else:
        st.markdown('在每个 OOS 折上计算 precision/recall/f1/accuracy/balanced_accuracy（宏平均或整体），此处展示聚合后的中位/均值。')
        cols_pred = [c for c in ['candidate','folds','f1_median','precision_median','recall_median','accuracy_median','balanced_accuracy_median'] if c in summary.columns]
        st.dataframe(summary[cols_pred].sort_values('f1_median', ascending=False), width='stretch')

        st.subheader('候选每折评估明细')
        cands = list_candidates(base_dir) if 'list_candidates' in globals() else []

        _qp_cand = _qp_get('cand', '')
        _cand_idx = cands.index(_qp_cand) if (_qp_cand in cands) else 0
        cand_sel = st.selectbox('选择候选查看每折明细', options=cands, index=_cand_idx if cands else 0, key='cand_sel')
        if cand_sel:
            _qp_set('cand', cand_sel)

        if cand_sel:
            df_fm = load_fold_metrics(base_dir, cand_sel)
            if df_fm.empty:
                st.warning(f'未找到 {os.path.join(base_dir, cand_sel, "fold_metrics.csv")}，请先重新执行回测生成明细。')
            else:
                # 可选：fold 日期区间映射（如果 run 导出了 fold_ranges.csv，就能精确显示每折 train/test 日期）
                df_ranges = load_fold_ranges(base_dir)
                if not df_ranges.empty and 'fold' in df_ranges.columns:
                    try:
                        df_fm = df_fm.merge(df_ranges, on='fold', how='left')
                    except Exception:
                        pass

                # 展示核心指标与类分布
                show_cols = [
                    'fold',
                    'precision','recall','f1','accuracy','balanced_accuracy',
                    'y_true_-1','y_true_0','y_true_1','y_pred_-1','y_pred_0','y_pred_1',
                    'train_start','train_end','test_start','test_end'
                ]
                cols_present = [c for c in show_cols if c in df_fm.columns]
                st.dataframe(df_fm[cols_present], width='stretch')

                st.divider()
                st.subheader('Fold 指标趋势')

                # 1) fold vs f1/accuracy 折线
                metric_opts = [m for m in ['f1','balanced_accuracy','accuracy','precision','recall'] if m in df_fm.columns]
                base_metric = 'f1' if 'f1' in metric_opts else (metric_opts[0] if metric_opts else None)
                chosen_metrics = st.multiselect('选择要画的指标（折线）', options=metric_opts, default=[base_metric] if base_metric else [])

                if chosen_metrics:
                    df_long = df_fm[['fold'] + chosen_metrics].melt(id_vars=['fold'], var_name='metric', value_name='value')
                    fig_line = px.line(df_long, x='fold', y='value', color='metric', markers=True, title=f'{cand_sel}：fold 指标趋势')
                    st.plotly_chart(fig_line, width='stretch')
                else:
                    st.caption('该 run 的 fold_metrics.csv 未包含可用指标列（f1/accuracy 等）。')

                # 2) 每 fold 类别分布（train / true / pred）
                st.subheader('每 fold 类别分布（Train / True / Pred）')
                colA, colB, colC = st.columns(3)

                def _plot_fold_class_dist(df_in: pd.DataFrame, prefix: str, title: str):
                    cols = [f'{prefix}_-1', f'{prefix}_0', f'{prefix}_1']
                    cols = [c for c in cols if c in df_in.columns]
                    if len(cols) != 3:
                        return None
                    tmp = df_in[['fold'] + cols].copy()
                    tmp = tmp.rename(columns={
                        f'{prefix}_-1': '-1',
                        f'{prefix}_0': '0',
                        f'{prefix}_1': '1',
                    })
                    long = tmp.melt(id_vars=['fold'], var_name='label', value_name='count')
                    fig = px.bar(long, x='fold', y='count', color='label', barmode='stack', title=title)
                    return fig

                with colA:
                    fig_train = _plot_fold_class_dist(df_fm, 'y_train', 'Train 标签分布（按 fold，堆叠）')
                    if fig_train is not None:
                        st.plotly_chart(fig_train, width='stretch')
                    else:
                        st.caption('fold_metrics.csv 未包含 y_train_-1/y_train_0/y_train_1 列（需要重新跑回测生成）。')

                with colB:
                    fig_true = _plot_fold_class_dist(df_fm, 'y_true', 'OOS True 标签分布（按 fold，堆叠）')
                    if fig_true is not None:
                        st.plotly_chart(fig_true, width='stretch')
                    else:
                        st.caption('fold_metrics.csv 未包含 y_true_-1/y_true_0/y_true_1 列，无法绘制 True 分布。')

                with colC:
                    fig_pred = _plot_fold_class_dist(df_fm, 'y_pred', 'OOS Pred 标签分布（按 fold，堆叠）')
                    if fig_pred is not None:
                        st.plotly_chart(fig_pred, width='stretch')
                    else:
                        st.caption('fold_metrics.csv 未包含 y_pred_-1/y_pred_0/y_pred_1 列，无法绘制 Pred 分布。')

                # 3) fold -> 日期区间（如果可用）
                st.subheader('Fold 对应日期区间')
                if all(c in df_fm.columns for c in ['train_start','train_end','test_start','test_end']):
                    st.caption('日期来自 run 目录中的 fold_ranges.csv（若不存在则此处为空）。')
                    st.dataframe(
                        df_fm[['fold','train_start','train_end','test_start','test_end']].sort_values('fold'),
                        width='stretch'
                    )
                else:
                    st.caption('该 run 未提供 fold_ranges.csv（或列缺失），因此无法映射每 fold 的 train/test 日期区间。')

# 候选对比（TopN对比：仅分类指标）
with tabs[3]:
    st.header('候选对比（预测评估）')
    if summary.empty:
        st.info('暂无 summary 数据')
    else:
        top_n_cmp = st.number_input('Top N（按 f1_median）', min_value=1, max_value=20, value=5, key='top_n_cmp_main')
        df_top_cmp = summary.sort_values('f1_median', ascending=False).head(top_n_cmp)
        cols_cmp = [c for c in ['candidate','T','X','folds','f1_median','precision_median','recall_median','accuracy_median','balanced_accuracy_median'] if c in df_top_cmp.columns]
        st.dataframe(df_top_cmp[cols_cmp], width='stretch')
        # 对比图：F1 vs Balanced Accuracy（示例）
        if all(k in df_top_cmp.columns for k in ['f1_median','balanced_accuracy_median']):
            fig_cmp = px.scatter(df_top_cmp, x='f1_median', y='balanced_accuracy_median', hover_name='candidate', title='F1 vs Balanced Accuracy (TopN)')
            st.plotly_chart(fig_cmp, width='stretch')

        # ===== 实验矩阵：多 run 对比 =====
        st.divider()
        st.subheader('实验矩阵：多 run 指标对比')
        st.caption('从各 run 的 summary CSV 读取指标，按 candidate 进行并集/交集对齐，并计算相对基准 run 的 delta。')

        if not run_names:
            st.info('未发现可用 runs（outputs/backtest/* 下需包含 reversal_param_search_wf_results.csv）')
        else:
            colx = st.columns([2, 2, 2])
            with colx[0]:
                # prefer a sensible default: baseline + flow if present
                _defaults = []
                for _k in ['baseline', 'flow']:
                    for _rn in run_names:
                        if _k in _rn and _rn not in _defaults:
                            _defaults.append(_rn)
                            break
                if len(_defaults) < 2:
                    _defaults = run_names[:2]

                _qp_runs = [r for r in _qp_get_list('runs') if r in run_names]
                runs_selected = st.multiselect('选择多个 run（建议 2~5 个）', options=run_names, default=_qp_runs or _defaults, key='runs_selected')
                if runs_selected:
                    _qp_set('runs', list(runs_selected))

            with colx[1]:
                metric_candidates = ['f1_median','balanced_accuracy_median','accuracy_median','precision_median','recall_median']
                _qp_sort = _qp_get('sort', 'f1_median')
                sort_by = st.selectbox('排序指标', options=metric_candidates, index=(metric_candidates.index(_qp_sort) if _qp_sort in metric_candidates else 0), key='sort_by')
                _qp_set('sort', sort_by)

            with colx[2]:
                base_opts = runs_selected if runs_selected else run_names
                _qp_base = _qp_get('base', '')
                base_idx = base_opts.index(_qp_base) if (_qp_base in base_opts) else 0
                base_run = st.selectbox('基准 run（delta = run - base）', options=base_opts, index=base_idx, key='base_run')
                _qp_set('base', base_run)

            mode_col = st.columns([1, 1, 1])
            with mode_col[0]:
                _qp_mode = _qp_get('cand_mode', '并集')
                cand_mode = st.radio('candidate 范围', options=['并集', '交集'], index=(0 if _qp_mode != '交集' else 1), horizontal=True, key='cand_mode')
                _qp_set('cand_mode', cand_mode)
            with mode_col[1]:
                _qp_pres = _safe_int(_qp_get('presence', '1'), 1)
                min_presence = st.number_input('至少出现在 N 个 run 中', min_value=1, max_value=max(1, len(runs_selected) if runs_selected else len(run_names)), value=_qp_pres, key='min_presence')
                _qp_set('presence', int(min_presence))
            with mode_col[2]:
                _qp_show = _safe_int(_qp_get('show_top', '50'), 50)
                show_top = st.number_input('展示 Top N（按基准 run 排序）', min_value=5, max_value=200, value=_qp_show, key='show_top')
                _qp_set('show_top', int(show_top))

            metrics = ['f1_median','balanced_accuracy_median','accuracy_median','precision_median','recall_median','folds']

            def _load_run_df(rn: str):
                d = load_summary(os.path.join(runs_root, rn))
                if d.empty:
                    return d
                keep = ['candidate'] + [m for m in metrics if m in d.columns]
                return d[keep].copy()

            if runs_selected and base_run in (runs_selected or []) and len(runs_selected) >= 2:
                dfs = {rn: _load_run_df(rn) for rn in runs_selected}
                cand_sets = {rn: set(df['candidate'].tolist()) for rn, df in dfs.items() if not df.empty and 'candidate' in df.columns}

                if not cand_sets:
                    st.warning('所选 runs 的 summary 为空，无法对比。')
                else:
                    if cand_mode == '交集':
                        candidates_set = set.intersection(*cand_sets.values())
                    else:
                        candidates_set = set.union(*cand_sets.values())

                    def _presence(c: str) -> int:
                        return sum(1 for s in cand_sets.values() if c in s)

                    candidates = [c for c in sorted(list(candidates_set)) if _presence(c) >= int(min_presence)]

                    if not candidates:
                        st.warning('没有满足条件的 candidate（可能是 min_presence 过高）。')
                    else:
                        rows = []
                        for cand in candidates:
                            row = {('','candidate'): cand, ('','presence'): _presence(cand)}
                            for rn in runs_selected:
                                df = dfs.get(rn)
                                if df is None or df.empty or 'candidate' not in df.columns:
                                    for m in metrics:
                                        row[(rn, m)] = pd.NA
                                    continue
                                rec = df[df['candidate'] == cand]
                                if rec.empty:
                                    for m in metrics:
                                        row[(rn, m)] = pd.NA
                                else:
                                    rec0 = rec.iloc[0]
                                    for m in metrics:
                                        row[(rn, m)] = rec0.get(m, pd.NA)
                            rows.append(row)

                        wide = pd.DataFrame(rows)
                        wide.columns = pd.MultiIndex.from_tuples(wide.columns, names=['run', 'metric'])

                        sort_key = (base_run, sort_by)
                        if sort_key in wide.columns:
                            wide = wide.sort_values(sort_key, ascending=False, na_position='last')

                        if show_top:
                            wide_show = wide.head(int(show_top))
                        else:
                            wide_show = wide

                        st.dataframe(wide_show, width='stretch')
            else:
                st.caption('请选择至少 2 个 run，并指定一个基准 run。')

# 文档
with tabs[4]:
    st.header('文档与说明')
    st.markdown('此处为使用说明文档，后续提供在线文档链接。')