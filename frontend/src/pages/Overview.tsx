import { Link } from "react-router-dom";
import { loadPerformance, loadWalkforward } from "../lib/data";
import {
  expandingPooled,
  fourModels,
  labelDelayRows,
  overallWindow,
  staticPooled,
} from "../lib/derived";
import { num } from "../lib/format";
import { useStatic } from "../lib/useStatic";
import LabelDelayBars from "../components/LabelDelayBars";
import ModelTable from "../components/ModelTable";
import StatusStrip from "../components/StatusStrip";
import { AsyncState, Card } from "../components/ui";

const PIPELINE = [
  {
    n: "01",
    title: "Real data, as a graph",
    body: "203,769 transactions and 822,942 wallets linked by 4.4 million money flows, over 49 two-week periods.",
    tech: "Elliptic++ · pandas",
  },
  {
    n: "02",
    title: "Four models compared",
    body: "A gradient-boosted tree model against three graph neural networks, all tested only on future data.",
    tech: "XGBoost · GraphSAGE · RGCN · HGT",
  },
  {
    n: "03",
    title: "A self-checking monitor",
    body: "Watches the model's own scores and confirmed labels, and raises a retrain alarm when they shift.",
    tech: "PSI drift · label-delay-safe rules",
  },
  {
    n: "04",
    title: "Live service",
    body: "The best model runs as an API that returns a fraud score and a reliability flag, tested on every push.",
    tech: "FastAPI · Docker · CI/CD",
  },
];

export default function Overview() {
  const perf = useStatic(loadPerformance);
  const wf = useStatic(loadWalkforward);

  if (perf.loading || wf.loading) return <AsyncState loading label="the dashboard data" />;
  if (perf.error || wf.error || !perf.data || !wf.data) {
    return <AsyncState loading={false} error={perf.error ?? wf.error} label="the dashboard data" />;
  }

  const p = perf.data;
  const w = wf.data;
  const baseline = overallWindow(p, "XGBoost", "35-49")?.pr_auc ?? null;
  const graphBest = overallWindow(p, "GraphSAGE", "35-49")?.pr_auc ?? null;
  const afterShift = overallWindow(p, "XGBoost", "43-49")?.pr_auc ?? null;
  const l1 = expandingPooled(w, 1, "43-49")?.pr_auc ?? null;
  const l3 = expandingPooled(w, 3, "43-49")?.pr_auc ?? null;

  const findings = [
    {
      title: "The simple model beat the graph networks",
      body: "XGBoost on transaction features ranked fraud far better than every graph neural network. Recent research on this dataset reports the same.",
      stat: `${num(baseline, 2)} vs ${num(graphBest, 2)}`,
      statLabel: "XGBoost vs best graph model",
      tone: "ink",
    },
    {
      title: "Every model broke when criminals changed behaviour",
      body: "After a dark-market shutdown around step 43, all four models fell to near-guessing. Most published work reports one average score and hides this.",
      stat: `${num(baseline, 2)} → ${num(afterShift, 2)}`,
      statLabel: "accuracy before vs after",
      tone: "drift",
    },
    {
      title: "Retraining fixes it, if fraud is confirmed fast",
      body: "Retraining on newly confirmed cases recovered most of the accuracy when labels arrived within one step. With a three-step delay, most of the gain was lost.",
      stat: `${num(l1, 2)} vs ${num(l3, 2)}`,
      statLabel: "1-step vs 3-step label delay",
      tone: "accent",
    },
  ];

  return (
    <div className="page page--landing">
      <section className="hero">
        <p className="eyebrow">Machine learning · fraud detection · MLOps</p>
        <h1 className="h1 h1--hero">
          A Bitcoin fraud detector that knows when to stop trusting itself.
        </h1>
        <p className="lede">
          BitcoinGraphGuard scores real Bitcoin transactions for signs of fraud, money laundering and dark-market
          activity. It also watches its own accuracy, raises an alarm when criminal behaviour changes, and is served as
          a live API.
        </p>
        <div className="hero__actions">
          <Link to="/try-it" className="btn btn--dark">
            Try a live prediction
          </Link>
          <a href="#findings" className="btn btn--ghost">
            See what I found
          </a>
        </div>
      </section>

      <section id="built" className="section" aria-labelledby="built-title">
        <h2 id="built-title" className="section__title">
          What I built
        </h2>
        <div className="pipeline">
          {PIPELINE.map((s) => (
            <div className="pipeline__card" key={s.n}>
              <span className="pipeline__n">{s.n}</span>
              <span className="pipeline__title">{s.title}</span>
              <span className="pipeline__body">{s.body}</span>
              <span className="pipeline__tech">{s.tech}</span>
            </div>
          ))}
        </div>
      </section>

      <section id="findings" className="section" aria-labelledby="findings-title">
        <h2 id="findings-title" className="section__title">
          What I found
        </h2>
        <div className="findings">
          {findings.map((f) => (
            <div className="finding" key={f.title}>
              <div className="finding__text">
                <span className="finding__title">{f.title}</span>
                <span className="finding__body">{f.body}</span>
              </div>
              <div className="finding__stat">
                <span className={`finding__value finding__value--${f.tone}`}>{f.stat}</span>
                <span className="finding__label">{f.statLabel}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="howto" aria-labelledby="howto-title">
        <h2 id="howto-title" className="howto__title">
          How to read the numbers
        </h2>
        <div className="howto__grid">
          <p>
            <strong>PR-AUC</strong> runs from 0 to 1 and measures how well the model ranks fraud above normal activity.
            Guessing scores about 0.07, the share of fraud in the data.
          </p>
          <p>
            <strong>Time step</strong> is about two weeks of blockchain activity. The models learn from steps 1–34 and
            are tested on the future, steps 35–49.
          </p>
          <p>
            <strong>Label delay</strong> is how long it takes investigators to confirm a transaction was fraud. The
            model can only learn from confirmed cases.
          </p>
        </div>
      </section>

      <section className="split">
        <Card className="card--wide">
          <div className="card-head">
            <h2 className="h2">Faster labels, better recovery</h2>
            <p className="subtle">
              Accuracy (PR-AUC) after the shift, by how many steps confirmed fraud labels take to arrive. Grey line is
              the 95% range.
            </p>
          </div>
          <LabelDelayBars rows={labelDelayRows(w)} staticRow={staticPooled(w)} />
          <Link to="/retraining" className="card__link">
            How retraining was tested →
          </Link>
        </Card>

        <Card className="card--narrow">
          <div className="card-head">
            <h2 className="h2">Four models compared</h2>
            <p className="subtle">Accuracy (PR-AUC) before and after the shift</p>
          </div>
          <ModelTable models={fourModels(p)} />
          <p className="footnote">Before = steps 35–42. After = steps 43–49.</p>
        </Card>
      </section>

      <StatusStrip />

      <footer className="site-footer">
        <span>Data: Elliptic++, real Bitcoin transactions with fraud labels. Tested on one real drift event.</span>
        <span>Python · XGBoost · PyTorch Geometric · FastAPI · Docker · GitHub Actions · React</span>
      </footer>
    </div>
  );
}
