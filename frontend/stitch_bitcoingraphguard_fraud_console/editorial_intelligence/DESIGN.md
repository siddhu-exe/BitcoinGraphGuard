---
name: Editorial Intelligence
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#3e4947'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#6e7977'
  outline-variant: '#bdc9c6'
  surface-tint: '#006a63'
  primary: '#005c55'
  on-primary: '#ffffff'
  primary-container: '#0f766e'
  on-primary-container: '#a3faef'
  inverse-primary: '#80d5cb'
  secondary: '#4648d4'
  on-secondary: '#ffffff'
  secondary-container: '#6063ee'
  on-secondary-container: '#fffbff'
  tertiary: '#7d4200'
  on-tertiary: '#ffffff'
  tertiary-container: '#a15600'
  on-tertiary-container: '#ffe6d5'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#9cf2e8'
  primary-fixed-dim: '#80d5cb'
  on-primary-fixed: '#00201d'
  on-primary-fixed-variant: '#00504a'
  secondary-fixed: '#e1e0ff'
  secondary-fixed-dim: '#c0c1ff'
  on-secondary-fixed: '#07006c'
  on-secondary-fixed-variant: '#2f2ebe'
  tertiary-fixed: '#ffdcc3'
  tertiary-fixed-dim: '#ffb77d'
  on-tertiary-fixed: '#2f1500'
  on-tertiary-fixed-variant: '#6e3900'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.03em
  headline-xl-mobile:
    fontFamily: Space Grotesk
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Space Grotesk
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  body-md:
    fontFamily: Space Grotesk
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  body-sm:
    fontFamily: Space Grotesk
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: -0.01em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 10px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-compact: 0.5rem
  margin: 1.5rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system deliberately departs from the ubiquitous sci-fi tropes of crypto tooling—eliminating aggressive neon greens, deep OLED black voids, and chaotic faux-hacker terminals. Instead, it frames institutional Bitcoin telemetry and on-chain graph analysis through the lens of modern financial journalism, Swiss modernist layout precision, and high-craft SaaS architectures like Stripe Sigma, Linear, and Vercel Analytics.

The design language fuses **Minimalism** with **High-Density Precision Engineering**:
- **Clarity over Sensationalism:** Off-white porcelain surfaces, paper-like matte structures, and rigorous structural hairline partitions evoke authority, calm rigor, and scientific auditability.
- **Micro-Contrast & Editorial Structure:** Dense data views rely on architectural grid dividers, deliberate hierarchy, and nuanced pastel status tags rather than high-contrast visual noise.
- **Tactile Restraint:** Depth is realized not through dramatic drop shadows, but through subtle layered offset borders, translucent porcelain planes, and micrometric hover elevations.

## Colors

The palette establishes an airy, light-refracting environment using layered porcelain values and muted, highly legible accents:

### Foundation Surfaces
- **Canvas Base (`#F8FAFC`):** Soft, neutral slate canvas that absorbs glare while providing superior contrast against white cards.
- **Card & Inset Surface (`#FFFFFF`):** High-clarity white for active graph viewports, data grids, and inspection panels.
- **Sub-surface / Structural Wells (`#F1F5F9`):** Recessed containers, table header bars, and utility rails.

### Structure & Hairline Dividers
- **Hairline Border (`#E2E8F0`):** Default structural line used for table borders, panel separators, and modular grid cells.
- **Active / Accent Border (`#CBD5E1`):** Applied on active item focus states, splitters, and pinned inspectors.

### Semantic Status & Metric System
- **Primary / Telemetry Core:** Deep Teal (`#0F766E`) against bright mint seafoam (`#2DD4BF`), supported by a soft tint background (`#CCFBF1`) for nominal status pills and active selection indicators.
- **Auxiliary Telemetry:** Indigo Periwinkle (`#6366F1`) on an ethereal lavender tint (`#EEF2FF`) for network clustering, routing heuristics, and secondary nodes.
- **Nominal / Settled:** Soft Pine Olive (`#16A34A`) on mint wash (`#DCFCE7`) for validated blocks, synced nodes, and verified graph paths.
- **Warning / Degraded / Recalibrate:** Muted Honey Amber (`#D97706`) on light amber tint (`#FEF3C7`) for mempool congestion, slow validation, and reorg notices.
- **Critical Drift / Alert:** Blush Coral Rose (`#E11D48` / `#F43F5E`) on soft blush wash (`#FFE4E6`) for anomalous script executions, double-spend warnings, and graph partition alarms.

### Typography Ink
- **Primary Ink (`#0F172A`):** Deep charcoal slate delivering sharp reading contrast.
- **Secondary Ink (`#334155`):** Muted slate for metadata keys, table headers, and structural labels.
- **Muted Ink (`#64748B`):** Ghost slate for disabled states, micro-hashes, and grid tick values.

## Typography

The typographic engine uses **Space Grotesk** for structural clarity and contemporary editorial polish, paired with **JetBrains Mono** for numerical precision and analytical rigor.

### Hierarchy & Typesetting Rules
- **Display & Headings:** Set tight letter-spacing on Space Grotesk (`-0.03em` on primary headers) to remove generic geometric bloat and deliver a sharp, technical presence.
- **Tabular Numerics & Telemetry (`JetBrains Mono`):** All hashes, transaction IDs, satoshi calculations, block heights, and timestamps must employ `font-variant-numeric: tabular-nums` to eliminate jitter during real-time streaming updates.
- **Data Density:** Body copy is scaled to compact 13px baseline metrics (`body-md`), ensuring analytical dashboards retain high information density without sacrificing optical comfort.

## Layout & Spacing

The layout is built upon an uncompromising **4px/8px baseline grid** supporting high-density workspace configurations:

### Grid Architecture
- **Desktop (>= 1280px):** Multi-pane split screen with a permanent 240px utility sidebar, 12-column variable workspace, and an optional 360px contextual telemetry inspector. Gutter is locked to `1rem` (16px) with canvas margins at `1.5rem` (24px).
- **Tablet (768px - 1279px):** 8-column layout. The inspector transitions to a collapsible slide-over drawer; gutter remains `1rem`.
- **Mobile (< 768px):** 4-column layout with `0.5rem` (8px) gutters and `1rem` margins. Complex multi-node graphs drop into condensed transactional cards and virtualized vertical list feeds.

### Panel Spacing & Layout Rhythm
- Inner component padding uses `space-sm` (8px) for table cells and chips, `space-md` (12px) for form inputs and control clusters, and `space-lg` (20px) for analytics card containers.
- Dense data clusters (such as node coordinate metrics or script-tree breadcrumbs) use `space-xs` (4px) gaps to reinforce visual chunking.

## Elevation & Depth

This design system avoids heavy shadows, instead using **crisp structural borders, subtle surface tone stacking, and micro-diffused ambient lift**:

- **Tier 0 (Canvas):** `#F8FAFC` base layer.
- **Tier 1 (Panels & Data Tables):** Solid `#FFFFFF` enclosed by a 1px solid `#E2E8F0` hairline border with a resting ambient shadow: `0 1px 2px 0 rgba(15, 23, 42, 0.04)`.
- **Tier 2 (Floating Popovers, Filter Menus & Node Tooltips):** `#FFFFFF` backing, 1px `#CBD5E1` border, accompanied by a double-blur shadow: `0 4px 6px -1px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04)`.
- **Tier 3 (Modals & Node Detail Flyouts):** High-precision layering with `0 20px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.04)`. Backdrops use `#0F172A` with `15%` opacity combined with a `4px` backdrop blur for optical grounding.
- **Active Node Highlight:** Instead of glowing halos, focused graph nodes and table rows use an interior `1px` ring in `#0F766E` paired with `#CCFBF1` pastel edge illumination.

## Shapes

The shape system utilizes **Soft (`1`) geometry** to uphold an architectural, tool-like posture reminiscent of financial terminals and technical instrumentation:

- **Micro Components (Chips, Inputs, Buttons, Badges):** Standardized on `0.25rem` (4px) radii. Avoids toy-like circular elements, retaining strict tabular rhythm.
- **Panels & Inset Cards:** Standardized on `0.5rem` (8px) corner radii (`rounded-lg`), providing just enough contour to separate components from the background canvas without wasting viewport area.
- **Modals & Drawers:** Maximum curvature capped at `0.75rem` (12px) (`rounded-xl`).
- **Graph Nodes & Visual Markers:** Graph entity vertices use square-proportional shapes with `2px` corners or exact circular badges only when signifying physical wallet entities vs transaction outputs.

## Components

### Buttons
- **Primary:** Background `#0F766E`, text `#FFFFFF`, 4px radius, 1px solid transparent border. On hover, transitions to `#115E59` with a subtle elevation shift. Focus states reveal a 2px offset ring in `#2DD4BF`.
- **Secondary / Outline:** Background `#FFFFFF`, text `#0F172A`, 1px solid hairline border `#E2E8F0`. Hover triggers a `#F1F5F9` background and `#CBD5E1` border tint.
- **Tertiary / Destructive:** Pastel coral background `#FFE4E6`, text `#E11D48`, 1px solid `#F43F5E` (20% opacity).

### Status Chips & Telemetry Badges
- **Structure:** Monospaced (`JetBrains Mono`), 10px font size, uppercase, tracking `0.04em`.
- **Nominal Badge:** Background `#DCFCE7`, text `#16A34A`, border `1px solid rgba(22, 163, 74, 0.2)`. Contains a 4px circular leading indicator.
- **Degraded Badge:** Background `#FEF3C7`, text `#D97706`, border `1px solid rgba(217, 119, 6, 0.2)`.
- **Critical / Drift Badge:** Background `#FFE4E6`, text `#E11D48`, border `1px solid rgba(225, 29, 72, 0.2)`.

### Form Inputs & Query Fields
- **Container:** Background `#FFFFFF`, 1px solid hairline border `#CBD5E1`, 4px radius, padding `0.5rem 0.75rem`.
- **Typography:** Value input in `JetBrains Mono` 13px for transaction queries and addresses.
- **Focus State:** 1px solid `#0F766E` with an ambient glow of `0 0 0 3px rgba(45, 212, 191, 0.2)`.

### Selection Controls (Checkboxes & Radios)
- **Base:** 14px x 14px boxes, 2px radius for checkboxes, full circle for radios. 1px solid border `#CBD5E1`.
- **Checked State:** Fill `#0F766E`, displaying a clean geometric check or center dot in `#FFFFFF`.

### Analytical Cards & Data Tables
- **Cards:** Background `#FFFFFF`, 8px radius, 1px solid `#E2E8F0`. Header regions feature a distinct 1px bottom border separating contextual actions from the metric body.
- **Tables:** Alternating rows avoid striped fills; hierarchy is achieved strictly via 1px horizontal `#E2E8F0` dividing rules. Row hover triggers a subtle background shift to `#F8FAFC`. Headers use Space Grotesk 11px uppercase tracking `0.05em` in `#64748B`.

### Specialized Guard Components
- **Hash Explorer Inset:** Monospaced 11px string container with truncated mid-hash ellipses (`0x7f2c...88a1`), enclosed in a `#F1F5F9` well with an integrated micro-copy button.
- **Graph Inspector Drawer:** Anchored floating porcelain card with active node metric tiles, script path verification tags, and balance delta streams rendered in tabbed tabular monospace.