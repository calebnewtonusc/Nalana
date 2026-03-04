"use client";

import { useEffect, useRef } from "react";
import Nav from "@/components/nav";
import Waitlist from "@/components/waitlist";

const ACCENT = "#7C3AED";
const HUB_URL = "https://specialized-model-startups.vercel.app";

function useScrollReveal() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { el.classList.add("visible"); obs.unobserve(el); } },
      { threshold: 0.12 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return ref;
}

function SectionLabel({ label }: { label: string }) {
  const ref = useScrollReveal();
  return (
    <div ref={ref} className="reveal flex items-center gap-5 mb-12">
      <span className="text-xs font-semibold uppercase tracking-[0.18em] text-gray-400 shrink-0">{label}</span>
      <div className="flex-1 h-px bg-gray-100" />
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen bg-white text-[#0a0a0a] overflow-x-hidden">
      <Nav />

      {/* Hero */}
      <section className="relative min-h-screen flex flex-col justify-center px-6 pt-14 overflow-hidden">
        {/* Subtle grid background */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage: `radial-gradient(circle at 20% 30%, ${ACCENT}08 0%, transparent 50%), radial-gradient(circle at 80% 70%, ${ACCENT}06 0%, transparent 50%)`,
          }}
        />

        <div className="relative max-w-5xl mx-auto w-full py-20">
          {/* Status pill */}
          <div className="fade-up delay-0 mb-8">
            <span
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold border"
              style={{ color: ACCENT, borderColor: `${ACCENT}30`, backgroundColor: `${ACCENT}08` }}
            >
              <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: ACCENT }} />
              Training &middot; 18&times; A6000 &middot; ETA Q2 2026
            </span>
          </div>

          {/* Model name */}
          <h1 className="fade-up delay-1 text-[clamp(3.5rem,10vw,7rem)] font-bold leading-[0.92] tracking-tight mb-6">
            <span className="serif font-light italic" style={{ color: ACCENT }}>Na</span>
            <span>lana</span>
          </h1>

          {/* Tagline */}
          <p className="fade-up delay-2 serif text-[clamp(1.25rem,3vw,2rem)] font-light text-gray-500 mb-4 max-w-xl">
            You speak. Nalana builds.
          </p>

          {/* Differentiator */}
          <p className="fade-up delay-3 text-sm text-gray-400 leading-relaxed max-w-lg mb-10">
            The only model trained on execution-verified 3D workflows across 9+ software platforms via a Universal DSL&nbsp;— not prompts, but verified scene diffs.
          </p>

          {/* Waitlist */}
          <div className="fade-up delay-4">
            <Waitlist />
          </div>
        </div>
      </section>

      {/* The Problem */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="The Problem" />
        <div className="grid md:grid-cols-2 gap-6">
          {/* General models */}
          <div className="reveal rounded-2xl border border-gray-100 p-8 bg-gray-50/50">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-5">What general models do</p>
            <ul className="space-y-3 text-sm text-gray-500 leading-relaxed">
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Generate Blender scripts that fail silently
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Transforms applied, materials missing, exports broken
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                No feedback loop — the model never sees if it worked
              </li>
              <li className="flex gap-3">
                <span className="text-gray-300 mt-0.5">&#8212;</span>
                Platform-locked: Blender syntax fails in Maya or Houdini
              </li>
            </ul>
          </div>

          {/* What Nalana does */}
          <div
            className="reveal rounded-2xl border p-8"
            style={{ borderColor: `${ACCENT}25`, backgroundColor: `${ACCENT}05` }}
          >
            <p className="text-xs font-semibold uppercase tracking-widest mb-5" style={{ color: ACCENT }}>What Nalana does</p>
            <ul className="space-y-3 text-sm leading-relaxed text-gray-700">
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Executes each command in a headless Blender sandbox
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Diffs the scene state before and after every step
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Retries automatically until the result matches intent
              </li>
              <li className="flex gap-3">
                <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
                Universal DSL translates across 9+ 3D platforms
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="How it works" />
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                step: "01",
                title: "Supervised Fine-Tuning",
                desc: "1.23M execution-verified (input, Universal DSL command, scene diff) triplets. Base: Qwen2.5-7B-Coder-Instruct. Nalana learns the mapping from natural language to verified 3D actions.",
              },
              {
                step: "02",
                title: "RL with Verifiable Reward",
                desc: "The reward is binary and unambiguous: headless Blender executes the generated command. Pass if the scene diff matches intent. Fail if it doesn't. No human labeling required.",
              },
              {
                step: "03",
                title: "DPO Alignment",
                desc: "Direct Preference Optimization on (correct execution, failed execution) pairs. Teaches Nalana to prefer topology-valid, physically-grounded, minimal outputs over verbose or incorrect ones.",
              },
            ].map(({ step, title, desc }) => {
              const ref = useScrollReveal();
              return (
                <div key={step} ref={ref} className="reveal-scale rounded-2xl border border-gray-100 bg-white p-8">
                  <div className="text-xs font-bold uppercase tracking-widest mb-4" style={{ color: ACCENT }}>{step}</div>
                  <h3 className="serif font-semibold text-lg mb-3 text-gray-900">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Capabilities */}
      <section className="px-6 py-24 max-w-5xl mx-auto">
        <SectionLabel label="Capabilities" />
        <div className="grid sm:grid-cols-2 gap-5">
          {[
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
                </svg>
              ),
              title: "Cross-platform 3D commands",
              desc: "Issue a single natural language command — Nalana translates to Blender, Maya, Cinema 4D, Houdini, Unreal Engine, or Rhino. One DSL, nine platforms.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
                </svg>
              ),
              title: "Physics-grounded material generation",
              desc: "Generate PBR materials, subsurface scattering, and index-of-refraction values derived from physics first principles — not texture atlas lookups.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/><polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/>
                </svg>
              ),
              title: "Multi-step workflow orchestration",
              desc: "Nalana executes multi-step pipelines — model, rig, shade, light, render — with auto-correction at each stage. Failures trigger retries, not silent errors.",
            },
            {
              icon: (
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                </svg>
              ),
              title: "Topology-aware mesh operations",
              desc: "Generates production-ready geometry with correct edge loops, manifold surfaces, and subdivision-ready topology — not just visually plausible meshes.",
            },
          ].map(({ icon, title, desc }) => {
            const ref = useScrollReveal();
            return (
              <div
                key={title}
                ref={ref}
                className="reveal rounded-2xl border border-gray-100 p-7 flex gap-5 hover:border-gray-200 transition-colors"
              >
                <div
                  className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ backgroundColor: `${ACCENT}10` }}
                >
                  {icon}
                </div>
                <div>
                  <h3 className="font-semibold text-sm text-gray-900 mb-1.5">{title}</h3>
                  <p className="text-sm text-gray-500 leading-relaxed">{desc}</p>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* The numbers */}
      <section className="px-6 py-24 bg-gray-50/50">
        <div className="max-w-5xl mx-auto">
          <SectionLabel label="The numbers" />
          <div className="grid sm:grid-cols-3 gap-6">
            {[
              { stat: "1.23M", label: "Training pairs", sub: "10k+ tutorials, 9 platforms" },
              { stat: "Qwen2.5-7B", label: "Base model", sub: "Coder-Instruct" },
              { stat: "Pass / Fail", label: "Reward signal", sub: "Headless Blender execution" },
            ].map(({ stat, label, sub }) => {
              const ref = useScrollReveal();
              return (
                <div
                  key={label}
                  ref={ref}
                  className="reveal rounded-2xl border p-8"
                  style={{ borderColor: `${ACCENT}20` }}
                >
                  <div className="text-3xl font-bold tracking-tight mb-2" style={{ color: ACCENT }}>{stat}</div>
                  <div className="text-sm font-semibold text-gray-800 mb-1">{label}</div>
                  <div className="text-xs text-gray-400">{sub}</div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 py-12 border-t border-gray-100">
        <div className="max-w-5xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-400">
          <p>
            Part of the{" "}
            <a href={HUB_URL} className="underline underline-offset-2 hover:text-gray-600 transition-colors">
              Specialist AI
            </a>{" "}
            portfolio by{" "}
            <a
              href="https://github.com/calebnewtonusc"
              target="_blank"
              rel="noopener noreferrer"
              className="underline underline-offset-2 hover:text-gray-600 transition-colors"
            >
              Caleb Newton &middot; calebnewtonusc
            </a>{" "}
            &middot; 2026
          </p>
        </div>
      </footer>
    </div>
  );
}
