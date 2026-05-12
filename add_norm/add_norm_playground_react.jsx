import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";

const dims = ["Dim 1", "Dim 2", "Dim 3", "Dim 4"];
const colors = ["bg-blue-500", "bg-emerald-500", "bg-amber-500", "bg-purple-500"];
const textColors = ["text-blue-600", "text-emerald-600", "text-amber-600", "text-purple-600"];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function mean(values) {
  return values.reduce((acc, v) => acc + v, 0) / values.length;
}

function variance(values, mu) {
  return values.reduce((acc, v) => acc + (v - mu) ** 2, 0) / values.length;
}

function addNorm(x, fx, gamma, beta, eps) {
  const z = x.map((v, i) => v + fx[i]);
  const mu = mean(z);
  const sigma2 = variance(z, mu);
  const std = Math.sqrt(sigma2 + eps);
  const norm = z.map((v, i) => gamma[i] * ((v - mu) / std) + beta[i]);
  return { z, mu, sigma2, std, norm };
}

function fmt(v) {
  return Number.isFinite(v) ? v.toFixed(3) : "0.000";
}

function VectorBars({ title, values, range = 3, showCenter = true }) {
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-semibold text-slate-800">{title}</h3>
        <span className="text-xs text-slate-500">feature vector</span>
      </div>

      <div className="space-y-3">
        {values.map((v, i) => {
          const left = ((clamp(v, -range, range) + range) / (range * 2)) * 100;
          const center = 50;
          const width = Math.abs(left - center);
          const barLeft = Math.min(left, center);

          return (
            <div key={i} className="grid grid-cols-[70px_1fr_70px] items-center gap-3">
              <div className={`text-sm font-medium ${textColors[i]}`}>{dims[i]}</div>
              <div className="relative h-7 rounded-full bg-slate-100">
                {showCenter && <div className="absolute left-1/2 top-0 h-7 w-px bg-slate-400" />}
                <motion.div
                  className={`absolute top-1 h-5 rounded-full ${colors[i]}`}
                  initial={false}
                  animate={{ left: `${barLeft}%`, width: `${Math.max(width, 1)}%` }}
                  transition={{ type: "spring", stiffness: 130, damping: 18 }}
                />
                <motion.div
                  className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white shadow ${colors[i]}`}
                  initial={false}
                  animate={{ left: `${left}%` }}
                  transition={{ type: "spring", stiffness: 130, damping: 18 }}
                />
              </div>
              <div className="text-right font-mono text-sm text-slate-700">{fmt(v)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SliderVector({ title, values, setValues, min = -2, max = 2, step = 0.05 }) {
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <h3 className="mb-3 font-semibold text-slate-800">{title}</h3>
      <div className="space-y-4">
        {values.map((v, i) => (
          <div key={i}>
            <div className="mb-1 flex justify-between text-sm">
              <span className={textColors[i]}>{dims[i]}</span>
              <span className="font-mono text-slate-700">{fmt(v)}</span>
            </div>
            <input
              className="w-full accent-blue-600"
              type="range"
              min={min}
              max={max}
              step={step}
              value={v}
              onChange={(e) => {
                const next = [...values];
                next[i] = Number(e.target.value);
                setValues(next);
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function StatCard({ label, value, desc }) {
  return (
    <motion.div
      className="rounded-2xl border bg-white p-4 shadow-sm"
      initial={false}
      animate={{ scale: [1, 1.02, 1] }}
      transition={{ duration: 0.25 }}
    >
      <div className="text-sm text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-2xl font-bold text-slate-900">{value}</div>
      <div className="mt-1 text-xs text-slate-500">{desc}</div>
    </motion.div>
  );
}

function FormulaPanel({ mu, sigma2, std, eps }) {
  return (
    <div className="rounded-2xl border bg-slate-950 p-5 text-white shadow-sm">
      <h3 className="mb-3 text-lg font-semibold">LayerNorm 계산식</h3>
      <div className="rounded-xl bg-white/10 p-4 font-mono text-sm leading-7">
        <div>z = x + F(x)</div>
        <div>μ = mean(z) = {fmt(mu)}</div>
        <div>σ² = mean((z - μ)²) = {fmt(sigma2)}</div>
        <div>std = sqrt(σ² + ε) = {fmt(std)}</div>
        <div className="mt-2 text-blue-200">LayerNorm(zᵢ) = γᵢ × ((zᵢ - μ) / std) + βᵢ</div>
      </div>
      <p className="mt-3 text-sm text-slate-300">
        ε = {eps}. 분산이 너무 작을 때 0으로 나누는 문제를 막기 위한 안정화 항이다.
      </p>
    </div>
  );
}

function FlowStep({ index, title, body, active }) {
  return (
    <motion.div
      className={`rounded-2xl border p-4 shadow-sm ${active ? "border-blue-500 bg-blue-50" : "bg-white"}`}
      initial={false}
      animate={{ y: active ? -4 : 0, scale: active ? 1.02 : 1 }}
      transition={{ type: "spring", stiffness: 160, damping: 18 }}
    >
      <div className="mb-2 flex items-center gap-2">
        <div className={`flex h-7 w-7 items-center justify-center rounded-full text-sm font-bold ${active ? "bg-blue-600 text-white" : "bg-slate-200 text-slate-700"}`}>
          {index}
        </div>
        <div className="font-semibold text-slate-800">{title}</div>
      </div>
      <div className="font-mono text-sm text-slate-700">{body}</div>
    </motion.div>
  );
}

export default function AddNormPlayground() {
  const [x, setX] = useState([0.95, 1.15, 0.2, 1.05]);
  const [fx, setFx] = useState([0.3, -0.1, 0.25, 0.05]);
  const [gamma, setGamma] = useState([1, 1, 1, 1]);
  const [beta, setBeta] = useState([0, 0, 0, 0]);
  const [eps, setEps] = useState(1e-5);
  const [step, setStep] = useState(4);

  const result = useMemo(() => addNorm(x, fx, gamma, beta, eps), [x, fx, gamma, beta, eps]);

  const vectorText = (arr) => `[${arr.map((v) => fmt(v)).join(", ")}]`;

  return (
    <main className="min-h-screen bg-slate-100 p-6 text-slate-900">
      <div className="mx-auto max-w-7xl space-y-6">
        <section className="rounded-3xl bg-white p-7 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-slate-950">Transformer Add & Norm Playground</h1>
              <p className="mt-2 max-w-3xl text-slate-600">
                C++ 예제의 <span className="font-mono">addNorm(input, sublayerOutput)</span> 부분을 프론트엔드에서 직접 조작하며 이해하는 학습용 UI다.
                마우스로 슬라이더를 움직이면 Residual Add와 LayerNorm 결과가 즉시 애니메이션으로 갱신된다.
              </p>
            </div>
            <div className="rounded-2xl bg-slate-950 px-5 py-4 text-white">
              <div className="text-sm text-slate-300">공식</div>
              <div className="font-mono text-sm">LayerNorm(z) = γ ⊙ ((z - μ) / √(σ² + ε)) + β</div>
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 gap-6 lg:grid-cols-[360px_1fr]">
          <aside className="space-y-6">
            <SliderVector title="입력 x" values={x} setValues={setX} />
            <SliderVector title="서브레이어 출력 F(x)" values={fx} setValues={setFx} />
            <SliderVector title="γ scale" values={gamma} setValues={setGamma} min={0} max={2} step={0.05} />
            <SliderVector title="β shift" values={beta} setValues={setBeta} min={-2} max={2} step={0.05} />

            <div className="rounded-2xl border bg-white p-4 shadow-sm">
              <div className="mb-1 flex justify-between text-sm">
                <span>epsilon ε</span>
                <span className="font-mono">{eps}</span>
              </div>
              <input
                className="w-full accent-blue-600"
                type="range"
                min="0.00001"
                max="0.1"
                step="0.00001"
                value={eps}
                onChange={(e) => setEps(Number(e.target.value))}
              />
            </div>
          </aside>

          <section className="space-y-6">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
              <FlowStep index="1" title="Input" body={`x = ${vectorText(x)}`} active={step === 1} />
              <FlowStep index="2" title="Sublayer" body={`F(x) = ${vectorText(fx)}`} active={step === 2} />
              <FlowStep index="3" title="Residual Add" body={`z = ${vectorText(result.z)}`} active={step === 3} />
              <FlowStep index="4" title="LayerNorm" body={`norm = ${vectorText(result.norm)}`} active={step === 4} />
            </div>

            <div className="rounded-3xl border bg-white p-5 shadow-sm">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <h2 className="text-xl font-bold">단계별 애니메이션</h2>
                <div className="flex gap-2">
                  {[1, 2, 3, 4].map((n) => (
                    <button
                      key={n}
                      onClick={() => setStep(n)}
                      className={`rounded-xl px-4 py-2 text-sm font-semibold transition ${step === n ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`}
                    >
                      STEP {n}
                    </button>
                  ))}
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                {(step >= 1) && <VectorBars title="1. 입력 x" values={x} />}
                {(step >= 2) && <VectorBars title="2. 서브레이어 출력 F(x)" values={fx} />}
                {(step >= 3) && <VectorBars title="3. Residual Add: z = x + F(x)" values={result.z} />}
                {(step >= 4) && <VectorBars title="4. LayerNorm(z): 중심 0, 스케일 안정화" values={result.norm} />}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <StatCard label="평균 μ" value={fmt(result.mu)} desc="z 벡터의 중심값" />
              <StatCard label="분산 σ²" value={fmt(result.sigma2)} desc="z 값들이 평균에서 퍼진 정도" />
              <StatCard label="표준편차 std" value={fmt(result.std)} desc="정규화에서 나누는 스케일" />
            </div>

            <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_420px]">
              <FormulaPanel mu={result.mu} sigma2={result.sigma2} std={result.std} eps={eps} />

              <div className="rounded-2xl border bg-white p-5 shadow-sm">
                <h3 className="mb-3 text-lg font-semibold">최종 출력 벡터</h3>
                <div className="rounded-2xl bg-slate-50 p-4 font-mono text-xl font-bold">
                  [
                  {result.norm.map((v, i) => (
                    <span key={i} className={textColors[i]}>
                      {i > 0 ? ", " : ""}{fmt(v)}
                    </span>
                  ))}
                  ]
                </div>
                <p className="mt-4 text-sm leading-6 text-slate-600">
                  이 값이 다음 Attention 또는 FFN으로 전달된다. Add만 하면 값의 중심과 크기가 계속 커질 수 있으므로,
                  LayerNorm이 각 토큰 내부 feature 차원 기준으로 분포를 다시 정렬한다.
                </p>
              </div>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}
