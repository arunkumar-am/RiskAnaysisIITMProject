import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Equity Fund Risk Analysis Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

BENCHMARK_OPTIONS = {"NIFTY 50": "NIFTY_50"}


class DashboardDataLoader:
    def __init__(self, data_dir='data', output_dir='output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def load_all_data(self):
        """Load all required data files"""
        try:
            data = {}

            # Load benchmark data
            benchmark_path = os.path.join(self.data_dir, 'benchmark_processed_NIFTY_50.csv')
            if os.path.exists(benchmark_path):
                data['nifty50'] = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)

            # Load fund data
            fund_path = os.path.join(self.data_dir, 'fund_processed_merged.csv')
            if os.path.exists(fund_path):
                data['funds'] = pd.read_csv(fund_path)
                data['funds']['date'] = pd.to_datetime(data['funds']['date'], format='%d-%m-%Y')
                data['funds'] = data['funds'].set_index('date')

            # Load output data without index
            output_files_no_index = [
                ('comparative', 'comparative_analysis.csv'),
                ('beta_alpha', 'beta_alpha_analysis.csv'),
                ('stress_test', 'stress_test_results.csv'),
                ('rolling_metrics', 'rolling_metrics_all_funds.csv'),
                ('volatility_analysis', 'volatility_analysis_all_funds.csv'),
                ('sentiment', 'sentiment_analysis.csv'),
            ]
            for key, filename in output_files_no_index:
                path = os.path.join(self.output_dir, filename)
                if os.path.exists(path):
                    data[key] = pd.read_csv(path)

            # Load output data with index (Fund as index)
            for key, filename in [('risk_metrics', 'risk_metrics_all_funds.csv'),
                                  ('risk_metrics_nifty50', 'risk_metrics_nifty50.csv')]:
                path = os.path.join(self.output_dir, filename)
                if os.path.exists(path):
                    data[key] = pd.read_csv(path, index_col=0)

            if os.path.exists(os.path.join(self.output_dir, 'ai_insights.json')):
                with open(os.path.join(self.output_dir, 'ai_insights.json'), 'r') as f:
                    data['ai_insights'] = json.load(f)

            if os.path.exists(os.path.join(self.output_dir, 'risk_narrative.txt')):
                with open(os.path.join(self.output_dir, 'risk_narrative.txt'), 'r') as f:
                    data['narrative'] = f.read()

            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return {}


class DashboardVisualizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _normalize_to_base(prices, base=100):
        if prices is None or len(prices) == 0:
            return prices
        return prices / prices.iloc[0] * base

    def plot_price_trends(self, fund_data, benchmark_data, fund_name, benchmark_name):
        """Plot normalized price trends for a single fund vs benchmark"""
        fig = go.Figure()

        fund_close = fund_data['Close'].dropna()
        bench_close = benchmark_data['Close'].dropna()
        common = fund_close.index.intersection(bench_close.index)

        fund_norm = self._normalize_to_base(fund_close.loc[common])
        bench_norm = self._normalize_to_base(bench_close.loc[common])

        fig.add_trace(go.Scatter(
            x=fund_norm.index,
            y=fund_norm,
            name=fund_name,
            line=dict(color='#1f77b4', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=bench_norm.index,
            y=bench_norm,
            name=benchmark_name,
            line=dict(color='#ff7f0e', width=2)
        ))

        fig.update_layout(
            title=f'Normalized Price Trends: {fund_name} vs {benchmark_name}',
            xaxis_title='Date',
            yaxis_title='Normalized Price (Base 100)',
            hovermode='x unified',
            height=400
        )

        return fig

    def plot_volatility_comparison(self, fund_data, benchmark_data, fund_name, benchmark_name):
        """Plot rolling volatility comparison"""
        fig = go.Figure()

        common = fund_data.index.intersection(benchmark_data.index)

        fig.add_trace(go.Scatter(
            x=fund_data.loc[common].index,
            y=fund_data.loc[common, 'Volatility_20D'],
            name=f'{fund_name} Volatility',
            line=dict(color='#1f77b4', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=benchmark_data.loc[common].index,
            y=benchmark_data.loc[common, 'Volatility_20D'],
            name=f'{benchmark_name} Volatility',
            line=dict(color='#ff7f0e', width=2)
        ))

        fig.update_layout(
            title='Rolling 20-Day Volatility Comparison',
            xaxis_title='Date',
            yaxis_title='Volatility',
            hovermode='x unified',
            height=400
        )

        return fig

    def plot_risk_metrics_comparison(self, fund_metrics, benchmark_metrics, fund_name, benchmark_name):
        """Plot risk metrics comparison"""
        metrics = ['Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'VaR_95']

        fund_values = [fund_metrics.get(m, np.nan) for m in metrics]
        benchmark_values = [benchmark_metrics.get(m, np.nan) for m in metrics]

        fig = go.Figure(data=[
            go.Bar(name=fund_name, x=metrics, y=fund_values),
            go.Bar(name=benchmark_name, x=metrics, y=benchmark_values)
        ])

        fig.update_layout(
            title='Risk Metrics Comparison',
            barmode='group',
            xaxis_title='Metrics',
            yaxis_title='Value',
            height=400
        )

        return fig

    def plot_sentiment_distribution(self, sentiment_data):
        """Plot sentiment distribution"""
        if sentiment_data is None or len(sentiment_data) == 0:
            return None

        sentiment_counts = sentiment_data['Sentiment_Label'].value_counts()

        fig = go.Figure(data=[
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6'])
            )
        ])

        fig.update_layout(
            title='Sentiment Distribution',
            height=400
        )

        return fig

    def plot_stress_test_results(self, stress_data, fund_name, benchmark_name='NIFTY 50'):
        """Plot stress test results"""
        if stress_data is None or len(stress_data) == 0:
            return None

        fig = go.Figure(data=[
            go.Bar(name=benchmark_name, x=stress_data['Scenario'], y=stress_data['LC_Drawdown_Impact']),
            go.Bar(name=fund_name, x=stress_data['Scenario'], y=stress_data['MC_Drawdown_Impact'])
        ])

        fig.update_layout(
            title=f'Stress Test: Drawdown Impact ({fund_name} vs {benchmark_name})',
            barmode='group',
            xaxis_title='Scenario',
            yaxis_title='Drawdown Impact',
            height=400,
            xaxis_tickangle=-45
        )

        return fig

    def plot_returns_distribution(self, fund_data, benchmark_data, fund_name, benchmark_name):
        """Plot returns distribution"""
        returns_fund = fund_data['Close'].pct_change().dropna()
        returns_bench = benchmark_data['Close'].pct_change().dropna()

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns_fund,
            name=f'{fund_name} Returns',
            opacity=0.7,
            nbinsx=50
        ))

        fig.add_trace(go.Histogram(
            x=returns_bench,
            name=f'{benchmark_name} Returns',
            opacity=0.7,
            nbinsx=50
        ))

        fig.update_layout(
            title='Daily Returns Distribution',
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            barmode='overlay',
            height=400
        )

        return fig

    def plot_drawdown_analysis(self, fund_data, benchmark_data, fund_name, benchmark_name):
        """Plot drawdown analysis"""
        def calculate_drawdown(prices):
            cummax = prices.cummax()
            drawdown = (prices - cummax) / cummax
            return drawdown

        dd_fund = calculate_drawdown(fund_data['Close'])
        dd_bench = calculate_drawdown(benchmark_data['Close'])
        common = dd_fund.index.intersection(dd_bench.index)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dd_fund.loc[common].index,
            y=dd_fund.loc[common],
            name=f'{fund_name} Drawdown',
            fill='tozeroy',
            line=dict(color='#1f77b4')
        ))

        fig.add_trace(go.Scatter(
            x=dd_bench.loc[common].index,
            y=dd_bench.loc[common],
            name=f'{benchmark_name} Drawdown',
            fill='tozeroy',
            line=dict(color='#ff7f0e')
        ))

        fig.update_layout(
            title='Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown',
            hovermode='x unified',
            height=400
        )

        return fig


def render_overview(data, visualizer, selected_fund, benchmark_name, benchmark_key):
    st.header("Overview")

    fund_metrics = None
    bench_metrics = None
    if 'risk_metrics' in data and selected_fund in data['risk_metrics'].index:
        fund_metrics = data['risk_metrics'].loc[selected_fund].to_dict()
    if 'risk_metrics_nifty50' in data and benchmark_key in data['risk_metrics_nifty50'].index:
        bench_metrics = data['risk_metrics_nifty50'].loc[benchmark_key].to_dict()

    if fund_metrics and bench_metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(f"{selected_fund} Volatility", f"{fund_metrics['Volatility']:.4f}")
        with col2:
            st.metric(f"{benchmark_name} Volatility", f"{bench_metrics['Volatility']:.4f}")
        with col3:
            st.metric("Volatility Ratio", f"{fund_metrics['Volatility'] / bench_metrics['Volatility']:.2f}")
        with col4:
            st.metric(f"{selected_fund} Sharpe", f"{fund_metrics['Sharpe_Ratio']:.4f}")

    if 'funds' in data and 'nifty50' in data:
        fund_df = data['funds'][data['funds']['Fund'] == selected_fund]
        if len(fund_df) > 0:
            fig = visualizer.plot_price_trends(fund_df, data['nifty50'], selected_fund, benchmark_name)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Selected fund data not available.")


def render_price_analysis(data, visualizer, selected_fund, benchmark_name):
    st.header("Price & Volatility Analysis")

    if 'funds' not in data or 'nifty50' not in data:
        st.info("Fund or benchmark data not available.")
        return

    fund_df = data['funds'][data['funds']['Fund'] == selected_fund]
    if len(fund_df) == 0:
        st.info("Selected fund data not available.")
        return

    benchmark_df = data['nifty50']

    col1, col2 = st.columns(2)
    with col1:
        fig = visualizer.plot_price_trends(fund_df, benchmark_df, selected_fund, benchmark_name)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = visualizer.plot_volatility_comparison(fund_df, benchmark_df, selected_fund, benchmark_name)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = visualizer.plot_returns_distribution(fund_df, benchmark_df, selected_fund, benchmark_name)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = visualizer.plot_drawdown_analysis(fund_df, benchmark_df, selected_fund, benchmark_name)
        st.plotly_chart(fig, use_container_width=True)


def render_risk_metrics(data, visualizer, selected_fund, benchmark_name, benchmark_key):
    st.header("Risk Metrics")

    if 'risk_metrics' in data:
        with st.expander("All Funds Risk Metrics"):
            st.dataframe(data['risk_metrics'], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if 'risk_metrics' in data and selected_fund in data['risk_metrics'].index:
            st.subheader(selected_fund)
            st.dataframe(data['risk_metrics'].loc[[selected_fund]], use_container_width=True)
        else:
            st.info("Selected fund risk metrics not available.")

    with col2:
        if 'risk_metrics_nifty50' in data and benchmark_key in data['risk_metrics_nifty50'].index:
            st.subheader(benchmark_name)
            st.dataframe(data['risk_metrics_nifty50'].loc[[benchmark_key]], use_container_width=True)
        else:
            st.info("Benchmark risk metrics not available.")

    fund_metrics = None
    bench_metrics = None
    if 'risk_metrics' in data and selected_fund in data['risk_metrics'].index:
        fund_metrics = data['risk_metrics'].loc[selected_fund].to_dict()
    if 'risk_metrics_nifty50' in data and benchmark_key in data['risk_metrics_nifty50'].index:
        bench_metrics = data['risk_metrics_nifty50'].loc[benchmark_key].to_dict()

    if fund_metrics and bench_metrics:
        fig = visualizer.plot_risk_metrics_comparison(fund_metrics, bench_metrics, selected_fund, benchmark_name)
        st.plotly_chart(fig, use_container_width=True)


def render_comparative(data, selected_fund, benchmark_name):
    st.header("Comparative Analysis")

    if 'comparative' not in data:
        st.info("Comparative data not available.")
        return

    comp = data['comparative'][data['comparative']['Fund'] == selected_fund]
    if len(comp) == 0:
        st.info("No comparative data for selected fund.")
        return

    st.dataframe(comp, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=benchmark_name,
        x=comp['Metric'],
        y=comp['Large-Cap']
    ))
    fig.add_trace(go.Bar(
        name=selected_fund,
        x=comp['Metric'],
        y=comp['Mid-Cap']
    ))
    fig.update_layout(
        title=f'Comparative Metrics: {selected_fund} vs {benchmark_name}',
        barmode='group',
        xaxis_title='Metric',
        yaxis_title='Value',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_beta_alpha(data, selected_fund):
    st.header("Beta & Alpha")

    if 'beta_alpha' in data:
        with st.expander("All Funds Beta & Alpha"):
            st.dataframe(data['beta_alpha'], use_container_width=True)

    if 'beta_alpha' in data and selected_fund in data['beta_alpha']['Fund'].values:
        fund_data = data['beta_alpha'][data['beta_alpha']['Fund'] == selected_fund].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Beta", f"{fund_data['Beta']:.4f}")
        with col2:
            st.metric("Alpha", f"{fund_data['Alpha']:.4f}")
        with col3:
            st.metric("Information Ratio", f"{fund_data['Information_Ratio']:.4f}")
    else:
        st.info("Beta & Alpha data not available for selected fund.")


def render_stress_testing(data, visualizer, selected_fund, benchmark_name):
    st.header("Stress Testing")

    if 'stress_test' not in data:
        st.info("Stress test data not available.")
        return

    stress = data['stress_test'][data['stress_test']['Fund'] == selected_fund]
    if len(stress) == 0:
        st.info("No stress test data for selected fund.")
        return

    fig = visualizer.plot_stress_test_results(stress, selected_fund, benchmark_name)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detailed Results"):
        st.dataframe(stress, use_container_width=True)


def render_sentiment(data, visualizer):
    st.header("Market Sentiment")

    if 'ai_insights' in data:
        insights = data['ai_insights']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Sentiment", f"{insights['Sentiment_Analysis']['Mean_Sentiment']:.4f}")
        with col2:
            st.metric("Sentiment Strength", f"{insights['Sentiment_Analysis']['Sentiment_Strength']:.4f}")
        with col3:
            st.metric("Positive Ratio", f"{insights['Sentiment_Analysis']['Positive_Ratio']:.1%}")
        with col4:
            st.metric("Negative Ratio", f"{insights['Sentiment_Analysis']['Negative_Ratio']:.1%}")

    if 'sentiment' in data:
        fig = visualizer.plot_sentiment_distribution(data['sentiment'])
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Sentiment Details"):
            st.dataframe(data['sentiment'], use_container_width=True)


def render_ai_insights(data):
    st.header("AI-Generated Insights")

    if 'ai_insights' in data:
        insights = data['ai_insights']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Volatility", f"{insights['Adjusted_Risk_Metrics']['Original_Volatility']:.4f}")
            st.metric("Original VaR (95%)", f"{insights['Adjusted_Risk_Metrics']['Original_VaR']:.4f}")
        with col2:
            st.metric("Adjusted Volatility", f"{insights['Adjusted_Risk_Metrics']['Adjusted_Volatility']:.4f}")
            st.metric("Adjusted VaR (95%)", f"{insights['Adjusted_Risk_Metrics']['Adjusted_VaR']:.4f}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Return", f"{insights['Monte_Carlo_Scenarios']['Mean_Return']:.4f}")
        with col2:
            st.metric("Std Return", f"{insights['Monte_Carlo_Scenarios']['Std_Return']:.4f}")
        with col3:
            st.metric("5th Percentile", f"{insights['Monte_Carlo_Scenarios']['Percentile_5']:.4f}")

    if 'narrative' in data:
        st.subheader("Risk Narrative & Recommendations")
        st.text(data['narrative'])


def main():
    st.title("AI-Powered Equity Fund Risk Analysis Dashboard")
    st.markdown("---")

    loader = DashboardDataLoader(data_dir='data', output_dir='output')
    data = loader.load_all_data()

    if not data:
        st.error("No data found. Please run main program.")
        return

    visualizer = DashboardVisualizer()

    # Determine available funds
    available_funds = set()
    if 'funds' in data and 'Fund' in data['funds'].columns:
        available_funds.update(data['funds']['Fund'].unique())
    if 'risk_metrics' in data:
        available_funds.update(data['risk_metrics'].index)
    if 'beta_alpha' in data and 'Fund' in data['beta_alpha'].columns:
        available_funds.update(data['beta_alpha']['Fund'].unique())

    available_funds = sorted([f for f in available_funds if pd.notna(f)])

    if not available_funds:
        st.error("No funds found in the loaded data.")
        return

    benchmark_name = list(BENCHMARK_OPTIONS.keys())[0]
    benchmark_key = BENCHMARK_OPTIONS[benchmark_name]

    with st.sidebar:
        st.header("Controls")

        selected_fund = st.selectbox("Select Fund", available_funds)
        st.caption(f"Benchmark: {benchmark_name}")

        selected_tab = st.radio(
            "Select View",
            [
                "Overview",
                "Price Analysis",
                "Risk Metrics",
                "Comparative Analysis",
                "Beta & Alpha",
                "Stress Testing",
                "Sentiment Analysis",
                "AI Insights"
            ]
        )

        st.markdown("---")
        st.markdown("Data Status")
        if 'nifty50' in data:
            st.success(f"NIFTY 50: {len(data['nifty50'])} records")
        if 'funds' in data:
            st.success(f"Funds: {data['funds']['Fund'].nunique()} unique")
        if 'risk_metrics' in data:
            st.success(f"Risk Metrics: {len(data['risk_metrics'])} funds")

    if selected_tab == "Overview":
        render_overview(data, visualizer, selected_fund, benchmark_name, benchmark_key)
    elif selected_tab == "Price Analysis":
        render_price_analysis(data, visualizer, selected_fund, benchmark_name)
    elif selected_tab == "Risk Metrics":
        render_risk_metrics(data, visualizer, selected_fund, benchmark_name, benchmark_key)
    elif selected_tab == "Comparative Analysis":
        render_comparative(data, selected_fund, benchmark_name)
    elif selected_tab == "Beta & Alpha":
        render_beta_alpha(data, selected_fund)
    elif selected_tab == "Stress Testing":
        render_stress_testing(data, visualizer, selected_fund, benchmark_name)
    elif selected_tab == "Sentiment Analysis":
        render_sentiment(data, visualizer)
    elif selected_tab == "AI Insights":
        render_ai_insights(data)


if __name__ == '__main__':
    main()
