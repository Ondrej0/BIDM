# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Week 1 companion notebook — first contact with marimo

    This is your first marimo notebook. Before you build anything for your
    project, take two minutes to understand **what you are looking at**:

    - A marimo notebook is a single **`.py` file** — plain Python, so it lives
      happily in Git and runs from top to bottom.
    - The page is made of **cells**. Each cell is one block of code, and marimo
      works out the dependencies between them automatically (the "reactive"
      part). Change one cell, and every cell that depends on it recomputes —
      there is no hidden state like a Jupyter kernel can develop.
    - Narrative goes in `mo.md(...)` cells; a result is simply the final
      expression of a code cell.

    This week we meet the module's running example, **Streamline Media**, a
    fictional UK streaming service whose BI team wants to predict subscriber
    churn. No analysis yet — we generate the data, look at it, and frame the
    business problem.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### How to run this notebook

    From the module materials folder:

    ```bash
    uv run notebooks/week-01-bi-dm-overview.py          # run headless
    uvx marimo edit notebooks/week-01-bi-dm-overview.py # explore interactively
    ```

    `uv run` executes the notebook as a script (no browser). `uvx marimo
    edit` opens it in the editor where you can see reactivity live. Try
    moving the slider below and watch the cells that depend on it recompute.
    """)
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(
        start=100,
        stop=3000,
        step=100,
        value=2000,
        label="Number of generated subscribers",
    )
    n_slider
    return (n_slider,)


@app.cell
def _(np, pd):
    def generate_streamline_data(n=2000, seed=42):
        """Synthetic Streamline Media subscriber dataset (reproducible)."""
        rng = np.random.default_rng(seed)
        tenure_months = rng.integers(1, 49, n)
        plan = rng.choice(["Basic", "Standard", "Premium"], n, p=[0.4, 0.4, 0.2])
        fee_map = {"Basic": 6.99, "Standard": 9.99, "Premium": 14.99}
        monthly_fee = np.array([fee_map[p] for p in plan])
        watch_hours = np.clip(
            rng.gamma(2.0, 6.0, n) - tenure_months * 0.05, 0, None
        )
        support_tickets = rng.poisson(0.6, n)
        satisfaction = np.clip(
            rng.normal(7, 2, n) - support_tickets * 0.4, 1, 10
        )
        logit = (
            0.5 + 0.45 * support_tickets + 0.12 * monthly_fee
            - 0.10 * watch_hours - 0.35 * satisfaction
        )
        churn_prob = 1 / (1 + np.exp(-logit))
        churned = rng.binomial(1, churn_prob)
        return pd.DataFrame(
            {
                "customer_id": [f"C{100000 + i}" for i in range(n)],
                "tenure_months": tenure_months,
                "plan": plan,
                "monthly_fee": monthly_fee,
                "watch_hours_month": np.round(watch_hours, 1),
                "support_tickets_90d": support_tickets,
                "satisfaction_score": np.round(satisfaction, 1),
                "region": rng.choice(
                    ["South West", "London", "Midlands", "Scotland"], n
                ),
                "churned": churned,
            }
        )

    return (generate_streamline_data,)


@app.cell
def _(generate_streamline_data, n_slider):
    df = generate_streamline_data(n=n_slider.value)
    return (df,)


@app.cell
def _(df, mo):
    mo.md(
        f"""
        ### First look: shape, head, churn rate

        The generated data has **{len(df):,}** rows and **{df.shape[1]}**
        columns — customer detail, plan, fee, engagement, support, region, and
        the target `churned`. The churn rate is **{df['churned'].mean():.1%}**,
        close to one in four subscribers. Move the slider above and every number
        on this page recomputes — that is reactivity in one sentence.
        """
    )
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(df, plt):
    plan_rates = df.groupby("plan")["churned"].mean().sort_values(ascending=False)
    overview_fig, overview_ax = plt.subplots(figsize=(5.4, 2.8))
    overview_ax.bar(plan_rates.index, plan_rates.values, color="#e30613")
    overview_ax.axhline(
        df["churned"].mean(),
        linestyle="--",
        color="#63666a",
        label="overall churn rate",
    )
    overview_ax.set_ylabel("churn rate")
    overview_ax.set_ylim(0, 0.4)
    overview_ax.set_title("Churn rate by plan (first glance)")
    overview_ax.legend(fontsize=8)
    overview_fig.tight_layout()
    overview_fig
    return


@app.cell
def _(df, mo):
    premium = df.loc[df["plan"] == "Premium", "churned"].mean()
    basic = df.loc[df["plan"] == "Basic", "churned"].mean()
    mo.md(
        f"""
        ### What we can already see

        This is not a full analysis — that begins in week 5. But a first glance
        surfaces the shape of the business problem: churn is not uniform across
        plans (Premium **{premium:.1%}** vs Basic **{basic:.1%}**). For
        Streamline Media that suggests *retention offers should be targeted at
        the plans that churn most*, and the value of a model is deciding *which*
        subscribers within those plans to prioritise.

        The ethical question arrives early, too. This is synthetic data, so no
        real people sit inside it — but a real churn model is built from real
        subscribers. Week 3 opens that properly; for now, just notice the
        difference between "predicting behaviour" and "profiling individuals".
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Your turn: section 1 of the main notebook

    This companion notebook is a **working notebook** — scratch space, never
    submitted. Your group's real work lives in the **main notebook**, which you
    start this week. Open it and write **section 1 — Project overview** as
    narrative:

    - a **problem statement**: what business problem are you investigating, and
      why does it matter?
    - **domain context**: the industry, the decision it supports, and who
      benefits
    - **group members** and roles
    - your **initial research questions** (2–3, not yet answered)
    - your **first literature search terms**
    - the **candidate datasets you scouted** (2–3 shortlisted, to be fully
      audited in week 4)

    Everything you assert here will be challenged in a viva — so build the habit
    from day one: no claim without a source you have actually read.
    """)
    return


if __name__ == "__main__":
    app.run()
