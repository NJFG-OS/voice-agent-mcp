# Tenant shutoff vs. the pod engine switch

Two different controls are often confused. They operate on different scopes, and only
one of them can ever affect another company.

| | **Suspend one company (billing / admin)** | **Pod engine switch (admin)** |
|---|---|---|
| Scope | One `CustomerAccount` | One `Workspace` (one pod) |
| Set by | NJFG CRM webhook, or an administrator | The admin engine panel |
| Field | `CustomerAccount.billing_hold` / `status` | `Workspace.engine_hard_off`, `voice_system_enabled` |
| Effect | That company's inbound routing and agent execution stop | That workspace's pod is powered down |
| Other companies | **Untouched** | **Untouched** |
| Reversible by | `/api/v1/client/resume` (billing hold) or an admin (manual suspension) | The admin switch, or the schedule |

## Suspending one company

`POST /api/v1/client/suspend` (CRM token) resolves **exactly one** profile from
`email`, `njfg_crm_id`, `workspace_id` or `stripe_customer_id`; identifiers that point
at more than one profile are refused with `409 ambiguous_client` rather than guessed.

`suspendClientForBilling()` (`src/lib/client-billing.server.js`) then:

1. writes `billing_hold: true`, `billing_hold_at`, `billing_hold_reason`,
   `billing_hold_invoice_id`, `status: "suspended"`, `onboarding_state: "suspended"`;
2. flips **that company's own number** to `inactive` in `TelephonyChannel` — the filter
   is `workspace_id` + the client's `twilio_phone_number`, so no other tenant's rows are
   touched.

Three independent gates then refuse that company's traffic:

- **Inbound routing** — `resolveInboundRoute()` returns `suspended: true` for the
  profile, so the call never reaches an agent (and the number is not re-routed to
  another tenant: the profile match wins before any channel or platform fallback).
- **Per-turn execution** — `billingHoldFor(workspace_id)` is checked in
  `runAgentTurn` and in `/api/pod/agent-config`, which returns the spoken
  `blocked_reason`. Both the client's voice workspace **and** its table workspace
  resolve to the same hold.
- **Pod config** — the relay asks the app for the workspace's config on every call, so
  a held workspace gets the blocked greeting instead of an agent.

What it deliberately does **not** do: it does not stop a pod, disable any other
workspace, change any other company's agents, or touch the shared runtime. A held
company on a pod shared with others simply answers with the hold message while the
other workspaces keep serving.

`POST /api/v1/client/resume` lifts **only a billing hold** and restores
`suspended_from_status`. A profile an administrator suspended by hand stays suspended
(`409 manual_suspension`), so an automated payment signal can never re-open an account
an operator closed.

## The pod engine switch

`masterOff()` / `masterOn()` (`src/lib/voice-control.server.js`) act on **one
workspace's pod**: they push `/session/end` scoped to that `workspace_id` and stop that
pod. `engine_hard_off` is the operator's explicit override and automatic callers
(schedule, market watch, inbound auto-arm) cannot lift it. Live calls require an
explicit `force`, and the session end carries the workspace id, so ending one tenant's
session never parks another tenant's armed profile.

In short: **suspending a company is data-level and per-company; the engine switch is
infrastructure-level and per-workspace.** Neither one is a lever that reaches across
companies.

## Latency, streaming and slot filling (current state)

- **Turn shape.** The relay asks for one or two short spoken sentences, plain text, and
  caps the completion at 220 tokens: a smaller cap is a shorter wait before the first
  syllable. The system prompt also forbids re-asking anything the caller already said
  in the call — the transcript is the only slot store, so a repeated question is a
  wasted turn and a second of silence.
- **Captured information.** Caller details live in the rolling transcript window
  (last 12 messages) plus the per-call workflow trace. There is no separate slot
  registry, so a workflow step that needs a value either relies on the transcript or
  sets `prompt_override` on its `agent_step` node.
- **Streaming.** The pipeline is turn-based, not token-streamed: transcribe → complete →
  synthesise → play, with barge-in clearing playback. The next latency win is
  sentence-level synthesis (start speaking the first sentence while the rest is still
  generating); it is a relay change, not an app change.
- **Barge-in** is already immediate: caller energy above the floor increments the
  generation counter, which stops the current playback and clears the Twilio buffer.
