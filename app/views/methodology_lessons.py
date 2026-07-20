import streamlit as st
from textwrap import dedent


def render_html(content: str) -> None:
    """Render HTML as one compact block so Markdown cannot create code blocks."""
    compact_html = " ".join(dedent(content).split())
    st.markdown(compact_html, unsafe_allow_html=True)


def render_methodology_lessons() -> None:
    """
    Render a comprehensive text-based project overview covering:

    - Project objective
    - End-to-end methodology
    - Completed development
    - Key modelling decisions
    - Technical achievements
    - Limitations
    - Lessons learned and reflection
    - Future research directions
    - Cloud deployment
    """

    render_html(
        """
        <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1600px;
        }

        .methodology-hero {
            background: linear-gradient(
                135deg,
                #F8FAFC 0%,
                #EFF6FF 100%
            );
            border: 1px solid #DCE7F5;
            border-radius: 18px;
            padding: 1.65rem 1.75rem;
            margin-bottom: 1.4rem;
        }

        .methodology-hero-title {
            color: #0F172A;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 0.55rem;
        }

        .methodology-hero-subtitle {
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.7;
            max-width: 1100px;
            margin: 0;
        }

        .section-heading {
            color: #0F172A;
            font-size: 1.35rem;
            font-weight: 800;
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        .section-subheading {
            color: #64748B;
            font-size: 0.91rem;
            line-height: 1.55;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
        }

        .content-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.25rem 1.3rem;
            height: 100%;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }

        .content-card-title {
            color: #0F172A;
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .content-card-text {
            color: #475569;
            font-size: 0.89rem;
            line-height: 1.65;
            margin: 0;
        }

        .content-card-text ul,
        .future-text ul {
            margin-top: 0.5rem;
            margin-bottom: 0;
            padding-left: 1.15rem;
        }

        .content-card-text li,
        .future-text li {
            margin-bottom: 0.42rem;
        }

        .workflow-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 1.15rem;
            height: 100%;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        }

        .workflow-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: #DBEAFE;
            color: #1D4ED8;
            font-weight: 800;
            font-size: 0.88rem;
            margin-bottom: 0.65rem;
        }

        .workflow-title {
            color: #0F172A;
            font-weight: 800;
            font-size: 0.97rem;
            margin-bottom: 0.4rem;
        }

        .workflow-description {
            color: #64748B;
            font-size: 0.85rem;
            line-height: 1.55;
        }

        .decision-card {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.8rem;
        }

        .decision-title {
            color: #0F172A;
            font-size: 0.95rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .decision-text {
            color: #475569;
            font-size: 0.87rem;
            line-height: 1.62;
        }

        .achievement-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 15px;
            padding: 1rem 1.05rem;
            height: 100%;
        }

        .achievement-label {
            color: #0F172A;
            font-weight: 800;
            font-size: 0.94rem;
            margin-bottom: 0.35rem;
        }

        .achievement-description {
            color: #64748B;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .limitation-card {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-radius: 16px;
            padding: 1.15rem 1.2rem;
            height: 100%;
        }

        .limitation-title {
            color: #92400E;
            font-size: 0.97rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .limitation-text {
            color: #78350F;
            font-size: 0.86rem;
            line-height: 1.62;
        }

        .lesson-card {
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 16px;
            padding: 1.2rem 1.25rem;
            height: 100%;
        }

        .lesson-title {
            color: #166534;
            font-size: 0.97rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .lesson-text {
            color: #166534;
            font-size: 0.86rem;
            line-height: 1.63;
        }

        .future-card {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 16px;
            padding: 1.2rem 1.25rem;
            height: 100%;
        }

        .future-title {
            color: #0F172A;
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .future-text {
            color: #475569;
            font-size: 0.87rem;
            line-height: 1.62;
        }

        .status-card {
            background: #EFF6FF;
            border: 1px solid #BFDBFE;
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
        }

        .status-title {
            color: #1E3A8A;
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 0.4rem;
        }

        .status-text {
            color: #1E40AF;
            font-size: 0.89rem;
            line-height: 1.65;
        }

        .reflection-card {
            background: #F8FAFC;
            border: 1px solid #CBD5E1;
            border-radius: 18px;
            padding: 1.4rem 1.5rem;
        }

        .reflection-text {
            color: #334155;
            font-size: 0.93rem;
            line-height: 1.78;
            margin: 0;
        }

        .technology-wrapper {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 0.8rem;
        }

        .technology-chip {
            background: #F1F5F9;
            border: 1px solid #CBD5E1;
            color: #334155;
            border-radius: 999px;
            padding: 0.4rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 650;
        }
        </style>
        """
    )

    # ================================================================
    # Hero
    # ================================================================

    render_html(
        """
        <div class="methodology-hero">
            <div class="methodology-hero-title">
                Methodology, Lessons & Future Research
            </div>

            <p class="methodology-hero-subtitle">
                A comprehensive overview of the design, development and
                evaluation of the Systematic ASX Equities Alpha Generation
                Platform. The project integrates market-data engineering,
                quantitative feature development, machine-learning modelling,
                walk-forward validation, long–short portfolio construction,
                backtesting and interactive cloud-based reporting within one
                end-to-end research workflow.
            </p>
        </div>
        """
    )

    # ================================================================
    # Project objective
    # ================================================================

    render_html(
        '<div class="section-heading">Project Objective</div>'
    )

    render_html(
        """
        <div class="content-card">
            <div class="content-card-text">
                The objective of this project was to investigate whether
                machine-learning models could convert stock-specific,
                industry-level and broader market information into
                economically meaningful cross-sectional equity signals.
                <br><br>

                Rather than assessing predictive accuracy in isolation, the
                platform was designed to evaluate the complete investment
                workflow: data collection, feature construction, model
                validation, stock ranking, portfolio formation, transaction
                costs and out-of-sample portfolio performance.
                <br><br>

                The resulting system constructs a weekly long–short portfolio
                across liquid ASX-listed equities. Stocks are ranked using
                predicted five-day returns, with the strongest forecasts
                allocated to the long portfolio and the weakest forecasts
                allocated to the short portfolio.
            </div>
        </div>
        """
    )

    # ================================================================
    # Methodology
    # ================================================================

    render_html(
        '<div class="section-heading">End-to-End Methodology</div>'
    )

    render_html(
        """
        <div class="section-subheading">
            The platform was developed as a modular pipeline, with each stage
            producing reusable datasets or outputs for subsequent components.
        </div>
        """
    )

    methodology_steps = [
        (
            "Data Acquisition",
            "Collected historical prices, trading volume, market index data, "
            "market-capitalisation information and company industry "
            "classifications for the ASX investment universe."
        ),
        (
            "Data Engineering",
            "Cleaned and aligned trading dates, calculated simple and "
            "logarithmic returns, generated industry datasets and stored "
            "reusable outputs in Parquet format."
        ),
        (
            "Feature Engineering",
            "Developed stock, market and industry features capturing "
            "momentum, volatility, relative performance, trend and "
            "mean-reversion behaviour."
        ),
        (
            "Model Development",
            "Implemented Decision Tree, LightGBM and XGBoost regression "
            "models to estimate each stock's future five-day return."
        ),
        (
            "Walk-Forward Validation",
            "Applied expanding-window training so every prediction used only "
            "information available before the relevant rebalance date."
        ),
        (
            "Portfolio Construction",
            "Ranked stocks by predicted return and converted model forecasts "
            "into weekly long and short portfolio positions."
        ),
        (
            "Backtesting",
            "Calculated realised portfolio returns using historical future "
            "returns and incorporated simplified transaction-cost "
            "assumptions."
        ),
        (
            "Interactive Reporting",
            "Built a Streamlit application presenting portfolio composition, "
            "model comparisons, backtest results and project documentation."
        )
    ]

    for start_index in range(0, len(methodology_steps), 4):
        row_steps = methodology_steps[start_index:start_index + 4]
        columns = st.columns(len(row_steps))

        for offset, (title, description) in enumerate(row_steps):
            with columns[offset]:
                render_html(
                    f"""
                    <div class="workflow-card">
                        <div class="workflow-number">
                            {start_index + offset + 1}
                        </div>

                        <div class="workflow-title">
                            {title}
                        </div>

                        <div class="workflow-description">
                            {description}
                        </div>
                    </div>
                    """
                )

        st.write("")

    # ================================================================
    # What was developed
    # ================================================================

    render_html(
        '<div class="section-heading">What Was Developed</div>'
    )

    developed_columns_1 = st.columns(2)

    with developed_columns_1[0]:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">
                    Data and Feature Pipeline
                </div>

                <div class="content-card-text">
                    <ul>
                        <li>
                            Automated historical data collection for ASX
                            equities and the ASX 200 market benchmark.
                        </li>

                        <li>
                            Reusable processing for prices, volume, returns,
                            logarithmic returns and market capitalisation.
                        </li>

                        <li>
                            Industry-level aggregation using
                            market-capitalisation-weighted returns.
                        </li>

                        <li>
                            Modular feature generation across stock, market
                            and industry information.
                        </li>

                        <li>
                            Parquet-based storage for faster and reproducible
                            analytical workflows.
                        </li>

                        <li>
                            Unit tests covering transformations, file creation
                            and disk-to-memory consistency.
                        </li>
                    </ul>
                </div>
            </div>
            """
        )

    with developed_columns_1[1]:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">
                    Modelling Pipeline
                </div>

                <div class="content-card-text">
                    <ul>
                        <li>
                            Common modelling interface for Decision Tree,
                            LightGBM and XGBoost regressors.
                        </li>

                        <li>
                            Expanding-window walk-forward training and
                            out-of-sample prediction.
                        </li>

                        <li>
                            Prediction-error evaluation using measures such as
                            MAE and MSE.
                        </li>

                        <li>
                            Cross-sectional evaluation using information
                            coefficient and directional hit rate.
                        </li>

                        <li>
                            Hyperparameter experimentation for the boosted
                            tree models.
                        </li>

                        <li>
                            Statistical hypothesis tests comparing aligned
                            model outcomes.
                        </li>
                    </ul>
                </div>
            </div>
            """
        )

    st.write("")

    developed_columns_2 = st.columns(2)

    with developed_columns_2[0]:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">
                    Portfolio and Backtesting Pipeline
                </div>

                <div class="content-card-text">
                    <ul>
                        <li>
                            Weekly stock ranking using predicted five-day
                            returns.
                        </li>

                        <li>
                            Long and short position selection from the upper
                            and lower prediction rankings.
                        </li>

                        <li>
                            Dollar-neutral portfolio construction.
                        </li>

                        <li>
                            Historical return calculation using realised
                            forward returns.
                        </li>

                        <li>
                            Simplified transaction-cost adjustment.
                        </li>

                        <li>
                            Evaluation using annual return, volatility, Sharpe,
                            Sortino, Calmar, drawdown and win-rate measures.
                        </li>
                    </ul>
                </div>
            </div>
            """
        )

    with developed_columns_2[1]:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">
                    Interactive Streamlit Application
                </div>

                <div class="content-card-text">
                    <ul>
                        <li>
                            Executive summary of the selected strategy.
                        </li>

                        <li>
                            Current portfolio holdings and exposure overview.
                        </li>

                        <li>
                            Backtest and benchmark performance presentation.
                        </li>

                        <li>
                            Model comparison and hypothesis-testing results.
                        </li>

                        <li>
                            Methodology, limitations and future-research
                            documentation.
                        </li>

                        <li>
                            Cloud deployment for external demonstration and
                            easier access.
                        </li>
                    </ul>
                </div>
            </div>
            """
        )

    # ================================================================
    # Features
    # ================================================================

    render_html(
        '<div class="section-heading">Feature and Signal Development</div>'
    )

    feature_columns = st.columns(3)

    with feature_columns[0]:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">
                    Stock-Specific Features
                </div>

                <div class="content-card-text">
                    Momentum, relative strength, rolling realised volatility,
                    volume-based indicators, trend estimates and
                    mean-reversion features were developed to capture
                    company-level return behaviour.
                </div>
            </div>
            """
        )

    with feature_columns[1]:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">
                    Market and Industry Features
                </div>

                <div class="content-card-text">
                    Market returns, industry-relative performance and broader
                    systematic information were incorporated to provide
                    context for each company's expected return.
                </div>
            </div>
            """
        )

    with feature_columns[2]:
        render_html(
            """
            <div class="content-card">
                <div class="content-card-title">
                    Experimental Research
                </div>

                <div class="content-card-text">
                    Additional research explored Kalman-filter trend signals,
                    Ornstein–Uhlenbeck mean reversion, residual-to-systematic
                    reversal, stock-pair relationships and interactions
                    between momentum and volume indicators.
                </div>
            </div>
            """
        )

    # ================================================================
    # Modelling decisions
    # ================================================================

    render_html(
        '<div class="section-heading">Key Modelling Decisions</div>'
    )

    render_html(
        """
        <div class="decision-card">
            <div class="decision-title">
                Why use walk-forward validation?
            </div>

            <div class="decision-text">
                Financial observations are ordered through time. A random
                train-test split could allow later market information to
                influence earlier predictions. Expanding-window validation
                preserves the chronological structure of the data and more
                closely resembles how the model would operate in practice.
            </div>
        </div>

        <div class="decision-card">
            <div class="decision-title">
                Why assess ranking quality as well as prediction error?
            </div>

            <div class="decision-text">
                The strategy is constructed from the relative ranking of
                stocks. A model may still support portfolio construction
                without forecasting the precise numerical value of every
                return. Information coefficient and directional hit rate
                therefore complement MAE and MSE.
            </div>
        </div>

        <div class="decision-card">
            <div class="decision-title">
                Why compare Decision Tree, LightGBM and XGBoost?
            </div>

            <div class="decision-text">
                The three models provide different levels of complexity while
                retaining the ability to capture nonlinear relationships and
                feature interactions. Their comparison tests whether more
                sophisticated boosting methods produce meaningful
                out-of-sample improvements over a transparent baseline.
            </div>
        </div>

        <div class="decision-card">
            <div class="decision-title">
                Why evaluate portfolio results separately?
            </div>

            <div class="decision-text">
                Predictive accuracy does not automatically translate into
                stronger investment outcomes. Portfolio performance also
                depends on ranking, position construction, diversification,
                turnover and transaction costs. Statistical model quality and
                economic outcomes were therefore evaluated separately.
            </div>
        </div>
        """
    )

    # ================================================================
    # Achievements
    # ================================================================

    render_html(
        '<div class="section-heading">Key Technical Achievements</div>'
    )

    achievements = [
        (
            "End-to-End Workflow",
            "Connected data collection, feature generation, modelling, "
            "portfolio construction, backtesting and reporting."
        ),
        (
            "Time-Aware Validation",
            "Implemented expanding-window evaluation to reduce look-ahead "
            "bias and support realistic out-of-sample testing."
        ),
        (
            "Modular Architecture",
            "Separated data, features, models, portfolio logic, metrics and "
            "dashboard components into reusable modules."
        ),
        (
            "Automated Testing",
            "Used pytest to validate transformations, saved datasets and "
            "pipeline consistency."
        ),
        (
            "Interactive Dashboard",
            "Presented portfolio, model and backtest results through an "
            "accessible Streamlit interface."
        ),
        (
            "Cloud Deployment",
            "Prepared the application for cloud-based access and external "
            "project demonstration."
        )
    ]

    achievement_columns = st.columns(3)

    for index, (title, description) in enumerate(achievements):
        with achievement_columns[index % 3]:
            render_html(
                f"""
                <div class="achievement-card">
                    <div class="achievement-label">
                        {title}
                    </div>

                    <div class="achievement-description">
                        {description}
                    </div>
                </div>
                """
            )

            if index < 3:
                st.write("")

    # ================================================================
    # Limitations
    # ================================================================

    render_html(
        '<div class="section-heading">Limitations</div>'
    )

    render_html(
        """
        <div class="section-subheading">
            These limitations should be considered when interpreting the
            reported backtest results. They mainly reflect data availability,
            practical assumptions and the scope of the research.
        </div>
        """
    )

    limitations = [
        (
            "Data Access",
            """
            The project relies primarily on publicly available market data.
            This limits access to institutional-quality information such as
            historical analyst forecasts, earnings revisions, point-in-time
            fundamentals, order-book data, securities-lending information and
            proprietary factor datasets.
            """
        ),
        (
            "Survivorship Bias",
            """
            The investment universe is based largely on companies available
            in current constituent datasets. Businesses that were delisted,
            acquired or removed during the historical period may be
            underrepresented, which could cause historical performance to
            appear stronger than a fully point-in-time backtest.
            """
        ),
        (
            "Time Constraints",
            """
            Development time was prioritised towards completing a functional
            end-to-end platform. The project therefore did not exhaustively
            investigate every possible feature, portfolio construction method,
            hyperparameter combination or alternative model architecture.
            """
        ),
        (
            "Simplified Transaction Costs",
            """
            Trading costs are currently represented using fixed basis-point
            assumptions. Actual costs vary across securities and time
            according to bid-ask spreads, liquidity, volatility, order size,
            market impact and short-borrow conditions.
            """
        ),
        (
            "Portfolio Construction Assumptions",
            """
            Portfolio construction currently uses relatively simple ranking
            and weighting rules. The framework does not yet optimise expected
            returns against turnover, covariance, liquidity, sector exposure
            or position-level constraints.
            """
        ),
        (
            "Market Scope",
            """
            The research focuses on daily-frequency Australian equities.
            Results may not generalise directly to international markets,
            alternative asset classes, intraday strategies or substantially
            different liquidity environments.
            """
        ),
        (
            "Changing Market Regimes",
            """
            The current approach applies broadly consistent modelling and
            portfolio rules across all market environments. Relationships
            learned during stable markets may weaken during volatility shocks,
            market declines or structural regime changes.
            """
        ),
        (
            "Short-Selling Assumptions",
            """
            The backtest assumes short positions can be established when
            required. It does not explicitly model stock-borrow availability,
            borrowing fees, recalls or short-selling restrictions.
            """
        ),
        (
            "Backtest Uncertainty",
            """
            Historical backtesting provides an estimate rather than a
            guarantee of future performance. Results remain sensitive to the
            selected period, feature definitions, rebalance rules, transaction
            costs and prevailing market conditions.
            """
        )
    ]

    for start_index in range(0, len(limitations), 3):
        limitation_row = limitations[start_index:start_index + 3]
        columns = st.columns(len(limitation_row))

        for column, (title, text) in zip(columns, limitation_row):
            with column:
                render_html(
                    f"""
                    <div class="limitation-card">
                        <div class="limitation-title">
                            {title}
                        </div>

                        <div class="limitation-text">
                            {text}
                        </div>
                    </div>
                    """
                )

        st.write("")

    # ================================================================
    # Lessons learned
    # ================================================================

    render_html(
        '<div class="section-heading">Lessons Learned</div>'
    )

    lessons = [
        (
            "Prediction error is only one component",
            """
            Lower model error does not necessarily produce a better portfolio.
            Ranking quality, position construction, turnover and transaction
            costs determine whether statistical predictions become
            economically useful.
            """
        ),
        (
            "Validation design matters",
            """
            A sophisticated model evaluated incorrectly may be less
            trustworthy than a simpler model tested through a realistic
            walk-forward framework. Preventing information leakage is
            fundamental in financial modelling.
            """
        ),
        (
            "Data quality constrains the research",
            """
            Missing histories, inconsistent fundamentals and current-universe
            datasets limit the conclusions that can be drawn. Higher-quality
            point-in-time data may add more value than simply increasing model
            complexity.
            """
        ),
        (
            "Complexity requires justification",
            """
            Decision Tree, LightGBM and XGBoost produced relatively similar
            outcomes in several comparisons. More sophisticated models should
            only be preferred when they demonstrate clear and consistent
            out-of-sample improvement.
            """
        ),
        (
            "Transaction costs can change conclusions",
            """
            Small predictive advantages may disappear when portfolio turnover
            and execution costs are included. Model evaluation should therefore
            extend beyond gross returns.
            """
        ),
        (
            "Software engineering improves research",
            """
            Modular code, reusable datasets, automated tests and consistent
            interfaces made experimentation more reliable and reduced the
            likelihood of silent implementation errors.
            """
        )
    ]

    for start_index in range(0, len(lessons), 2):
        lesson_row = lessons[start_index:start_index + 2]
        columns = st.columns(len(lesson_row))

        for column, (title, text) in zip(columns, lesson_row):
            with column:
                render_html(
                    f"""
                    <div class="lesson-card">
                        <div class="lesson-title">
                            {title}
                        </div>

                        <div class="lesson-text">
                            {text}
                        </div>
                    </div>
                    """
                )

        st.write("")

    # ================================================================
    # Future research
    # ================================================================

    render_html(
        '<div class="section-heading">Future Research Directions</div>'
    )

    render_html(
        """
        <div class="section-subheading">
            These areas represent natural extensions to the completed
            platform rather than requirements for the current version.
        </div>
        """
    )

    future_research = [
        (
            "Hidden Markov Models and Regime Detection",
            """
            Investigate Hidden Markov Models or related regime-switching
            methods to identify latent market environments such as stable
            growth, high volatility, market decline and recovery.

            <br><br>

            The estimated regime could be used as an additional feature or to
            dynamically adjust:

            <ul>
                <li>model selection or parameters,</li>
                <li>feature importance,</li>
                <li>position sizing,</li>
                <li>long and short exposure,</li>
                <li>turnover limits, and</li>
                <li>portfolio rebalancing rules.</li>
            </ul>
            """
        ),
        (
            "Dynamic Transaction-Cost Modelling",
            """
            Replace fixed basis-point costs with a stock-specific execution
            model that changes across time and market conditions.

            <br><br>

            Possible components include:

            <ul>
                <li>bid-ask spread estimates,</li>
                <li>average daily turnover,</li>
                <li>order participation rates,</li>
                <li>market impact,</li>
                <li>volatility-dependent execution costs,</li>
                <li>short-borrow fees, and</li>
                <li>turnover penalties.</li>
            </ul>
            """
        ),
        (
            "Point-in-Time Universe Construction",
            """
            Source historical ASX index membership, delisting and corporate
            action data to reconstruct the investable universe at each
            rebalance date. This would reduce survivorship bias and provide a
            more realistic estimate of historical strategy performance.
            """
        ),
        (
            "Turnover-Aware Portfolio Construction",
            """
            Extend the current ranking rules through position limits,
            turnover penalties, liquidity constraints, sector controls and
            volatility-adjusted position sizing. This would improve the link
            between model forecasts and implementable portfolio decisions.
            """
        ),
        (
            "Expanded Data and Features",
            """
            Explore point-in-time fundamentals, valuation ratios, earnings
            revisions, macroeconomic indicators and alternative datasets.
            Additional features would continue to be assessed using strict
            walk-forward validation.
            """
        ),
        (
            "Alternative Model Architectures",
            """
            Investigate ensemble methods, probabilistic forecasting,
            transformer-based time-series models and graph-based approaches.
            More complex models would only be retained where they demonstrate
            consistent economic improvement over simpler baselines.
            """
        ),
        (
            "Model Explainability and Monitoring",
            """
            Add SHAP-based explanations, feature-stability checks, prediction
            drift monitoring and rolling model diagnostics to identify why
            forecasts change and when learned relationships begin to weaken.
            """
        ),
        (
            "Paper Trading and Prospective Evaluation",
            """
            Evaluate the strategy prospectively using newly arriving data.
            Paper trading would test data timing, portfolio-generation logic,
            execution assumptions and realised transaction costs outside the
            historical backtest.
            """
        )
    ]

    for start_index in range(0, len(future_research), 2):
        future_row = future_research[start_index:start_index + 2]
        columns = st.columns(len(future_row))

        for column, (title, text) in zip(columns, future_row):
            with column:
                render_html(
                    f"""
                    <div class="future-card">
                        <div class="future-title">
                            {title}
                        </div>

                        <div class="future-text">
                            {text}
                        </div>
                    </div>
                    """
                )

        st.write("")

    # ================================================================
    # Deployment
    # ================================================================

    render_html(
        '<div class="section-heading">Deployment and Productionisation</div>'
    )

    render_html(
        """
        <div class="status-card">
            <div class="status-title">
                Cloud Deployment
            </div>

            <div class="status-text">
                The Streamlit application is being deployed to the cloud so
                the complete research platform can be accessed through an
                interactive web interface. This transforms the project from a
                local analytical workflow into a shareable demonstration of
                the underlying modelling and portfolio pipeline.
                <br><br>

                The codebase has been organised to support reproducible model
                runs, reusable data outputs and containerised execution.
                Future production work could introduce scheduled data
                refreshes, automated retraining, model-version tracking,
                application monitoring and prospective portfolio updates.
            </div>
        </div>
        """
    )

    # ================================================================
    # Reflection
    # ================================================================

    render_html(
        '<div class="section-heading">Project Reflection</div>'
    )

    render_html(
        """
        <div class="reflection-card">
            <p class="reflection-text">
                This project demonstrated that systematic equity modelling is
                not simply a model-selection exercise. The predictive model is
                only one component within a wider process involving data
                engineering, feature design, chronological validation,
                portfolio construction, backtesting and transaction costs.
                <br><br>

                One of the main findings was that additional model complexity
                did not automatically result in materially different portfolio
                outcomes. Decision Tree, LightGBM and XGBoost frequently
                produced similar prediction and portfolio results. This
                reinforced the value of transparent baseline models and the
                importance of requiring complex approaches to demonstrate
                clear out-of-sample improvements.
                <br><br>

                The project also highlighted the difference between an
                encouraging historical result and a fully robust investment
                process. Survivorship bias, restricted access to point-in-time
                data, simplified execution assumptions and changing market
                regimes may all affect the interpretation of backtested
                performance.
                <br><br>

                From an engineering perspective, modular pipelines, automated
                tests, reusable model interfaces and consistent stored
                datasets substantially improved research reproducibility. The
                resulting platform provides a strong foundation for future
                experimentation with market-regime detection, more realistic
                transaction-cost modelling and prospective paper-trading
                evaluation.
            </p>
        </div>
        """
    )

    # ================================================================
    # Technology stack
    # ================================================================

    render_html(
        '<div class="section-heading">Technology Stack</div>'
    )

    technologies = [
        "Python",
        "pandas",
        "NumPy",
        "scikit-learn",
        "LightGBM",
        "XGBoost",
        "statsmodels",
        "Plotly",
        "Streamlit",
        "pytest",
        "Parquet",
        "yfinance",
        "Git",
        "Docker",
        "AWS"
    ]

    technology_html = "".join(
        f'<span class="technology-chip">{technology}</span>'
        for technology in technologies
    )

    render_html(
        f"""
        <div class="content-card">
            <div class="content-card-text">
                The platform was developed using a Python-based quantitative
                and machine-learning stack, supported by automated testing,
                version control, containerisation and cloud-deployment tools.
            </div>

            <div class="technology-wrapper">
                {technology_html}
            </div>
        </div>
        """
    )