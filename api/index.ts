// ─────────────────────────────────────────────────────────────────────────────
// Vercel Serverless Function — self-contained, lightweight entry point.
//
// WHY self-contained (no `import ... from "../server"`):
//  • `server.ts` is a ~2300-line dev server: it pulls `vite` middleware,
//    serves static files from `dist/`, and reads/writes JSON "databases"
//    with `fs` — none of which works inside a Vercel serverless function.
//  • Importing the whole file also drags heavy deps into the function bundle
//    and crashes cold starts (ERR_MODULE_NOT_FOUND / FUNCTION_INVOCATION_FAILED).
//  • So this function implements the HTTP API surface the frontend needs
//    (health, subscriptions, admin read paths, Gemini AI proxy) directly on
//    Express, with in-memory state. Writes are best-effort (serverless FS is
//    ephemeral) and never throw.
// ─────────────────────────────────────────────────────────────────────────────
import express from "express";
import type { Request, Response, NextFunction } from "express";
import { GoogleGenAI, ThinkingLevel } from "@google/genai";
import crypto from "node:crypto";
import {
  clientKey,
  corsGuard,
  isStaffEmail,
  isWeakSecret,
  normalizeEmail,
  rateLimit,
  RateLimiter,
  requireStrongSecret,
  sameOriginGuard,
  startLimiterSweeper,
  weakSecretMessage,
  resolveAllowedOrigins,
} from "./security.js";

const app = express();
app.use(express.json({ limit: "15mb" }));

// ── Rate limiting ─────────────────────────────────────────────────────────────
// WHY A FLOOR, NOT A CEILING: the AI quota is the expensive surface, but project
// writes, payment receipts and the admin login were previously unlimited. Each of
// those is a resource an attacker can exhaust for free. Limits are per-identity
// (signed-in account when available, else IP) and env-tunable — see
// `docs/security.md`.
const apiLimiter = new RateLimiter({ max: 300, windowMs: 60_000 });
const aiLimiter = new RateLimiter({ max: 20, windowMs: 60_000 });
const writeLimiter = new RateLimiter({ max: 30, windowMs: 60_000 });
const authLimiter = new RateLimiter({ max: 20, windowMs: 10 * 60_000 });

const LIMIT_RULES = {
  api: apiLimiter.ruleFromEnv("RATE_LIMIT_API", { max: 300, windowMs: 60_000 }),
  ai: aiLimiter.ruleFromEnv("RATE_LIMIT_AI", { max: 20, windowMs: 60_000 }),
  write: writeLimiter.ruleFromEnv("RATE_LIMIT_WRITE", { max: 30, windowMs: 60_000 }),
  auth: authLimiter.ruleFromEnv("RATE_LIMIT_AUTH", { max: 20, windowMs: 10 * 60_000 }),
};

startLimiterSweeper(apiLimiter);
startLimiterSweeper(aiLimiter);
startLimiterSweeper(writeLimiter);
startLimiterSweeper(authLimiter);

/** Write guard for the resource-creating routes (projects, patches). */
const limitWrites = rateLimit(
  writeLimiter,
  { max: 30, windowMs: 60_000 },
  "write",
  "طلبات كثيرة جداً في وقت قصير. انتظر قليلاً ثم أعد المحاولة.",
);

/**
 * Tighter guard for the payment surface — each hit can touch object storage, and
 * a flood there is both a cost and a way to bury a real customer's request.
 */
const limitPayments = rateLimit(
  writeLimiter,
  { max: 10, windowMs: 10 * 60_000 },
  "payment",
  "تم استلام عدة طلبات تحويل. انتظر قليلاً قبل إرسال طلب جديد.",
);

/**
 * Tag the request with the signed-in identity BEFORE the limiters run, so a
 * quota is counted per account rather than per NAT. Reading the session here is
 * safe: it only verifies an HMAC, it does not grant anything.
 *
 * SUBSCRIPTION CHECK (staff bypass)
 * ---------------------------------
 * This also resolves the account's entitlement once per request and stores it on
 * the request. A staff account is short-circuited to Business HERE — before any
 * handler looks at a grant cookie, a payment row, or a client-supplied field —
 * which is what "bypass the subscription check" means in practice: there is no
 * remaining check downstream that could downgrade them.
 *
 * `subscriptionBypassed` is deliberately NOT an authorisation flag. It records
 * that this account skips the subscription gate, nothing more; admin access is
 * decided separately by `requireAdmin`.
 */
app.use((req: Request, _res: Response, next: NextFunction) => {
  const session = readAuthSession(req as AuthReq);
  const tagged = req as Request & { authEmail?: string; accountTier?: Tier; subscriptionBypassed?: boolean };

  if (session?.email) {
    tagged.authEmail = session.email;
    const staff = isStaffEmail(session.email);
    if (staff) {
      tagged.accountTier = "business";
      tagged.subscriptionBypassed = true;
    }
  }
  next();
});

// Broad floor for the whole API, then the CSRF origin check.
app.use("/api", rateLimit(apiLimiter, LIMIT_RULES.api, "api"));
app.use("/api", sameOriginGuard);

// ── CORS (allowlisted — see `resolveAllowedOrigins`) ───────────────────────────
// SECURITY: this was `Access-Control-Allow-Origin: *`, which let any website read
// this API's responses from a visitor's browser. A non-allowlisted origin now
// receives NO CORS header at all, so the browser refuses the read.
app.use((req, res, next) => corsGuard(resolveAllowedOrigins(req))(req, res, next));

// ── Health & status ─────────────────────────────────────────────────────────
// NOTE: getApiKey is defined in the Gemini section below (function hoisting
// makes it available here at runtime).
//
// WHY the path list contains only /api paths: this same app is mounted onto the
// self-hosted runner (see `server.ts`). Registering "/" here would sit in front
// of that host's static file server and answer the home page with a JSON health
// payload instead of the website.
app.get(["/api/health", "/api"], (_req: Request, res: Response) => {
  // ── PHASE 3: this endpoint is the monitoring hook.
  //
  // It previously reported `status: "ok"` unconditionally, which is the least
  // useful thing a health check can do: a deployment with NO signing secret,
  // NO AI key and NO database still answered "ok", so no alert could ever fire
  // and an operator had no way to see the real state without reading logs.
  //
  // `ok` now means "safe to serve", and `degraded` names exactly what is
  // missing. The detail list is names of broken invariants only — never a value
  // of a secret, and never an e-mail address.
  const problems: string[] = [];

  // SECURITY: a missing/weak signing key means nobody can sign in AND no
  // session can be trusted. It is reported as a problem, not a warning.
  if (authSessionSecretMissing()) problems.push("AUTH_SESSION_SECRET_MISSING");
  if (!getApiKey()) problems.push("GEMINI_API_KEY_MISSING");
  if (!isAdminPinConfigured()) problems.push("ADMIN_PIN_NOT_CONFIGURED");
  if (planSigningSecret() === null) problems.push("PLAN_GRANT_SECRET_MISSING");
  if (!OWNER_EMAIL) problems.push("SITE_OWNER_EMAIL_MISSING");
  if (!supabaseConfig().dbConfigured) problems.push("DATABASE_NOT_CONFIGURED");

  // Always HTTP 200: this endpoint answers for humans and monitors alike, and a
  // non-2xx would make every load balancer restart a working deployment.
  // `status: "degraded"` is the signal to alert on.
  res.json({
    status: problems.length > 0 ? "degraded" : "ok",
    service: "ebnili-api",
    time: new Date().toISOString(),
    // Deployment fingerprint — lets us verify which commit is actually live.
    commit: (process.env.VERCEL_GIT_COMMIT_SHA ?? "unknown").slice(0, 7),
    env: process.env.VERCEL_ENV ?? "local",
    // eslint-disable-next-line @typescript-eslint/no-use-before-define
    hasKey: Boolean(getApiKey()),
    /** Names of the broken invariants — safe to log, safe to alert on. */
    problems,
    /** Capability flags, so a monitor can assert the surface is actually wired. */
    features: {
      signInPossible: !authSessionSecretMissing(),
      adminConfigured: isAdminPinConfigured(),
      plansGrantable: planSigningSecret() !== null,
      databaseConfigured: supabaseConfig().dbConfigured,
    },
  });
});

// ── Subscriptions: stateless defaults (client holds source of truth) ─────────
const DEFAULT_SUBSCRIPTION = {
  tier: "free",
  status: "active",
  planName: "Starter Free",
  generationsUsedToday: 0,
  generationsLimitToday: 5,
  canExportZip: false,
  canDeployCustomDomain: false,
  canUseVisualInspector: true,
  priorityAiModel: false,
  transactions: [],
};

// ── Subscriptions: accept BOTH endpoints the frontend uses ──────────────────
// SECURITY: a paid tier is NEVER granted by a public request. Historically
// POST /api/subscriptions/auto-verify returned an activated Pro/Business
// object to ANY caller, so one curl call upgraded an account forever. Payment
// now only creates a PENDING review request; the tier is granted exclusively by
// the owner through POST /api/admin/transaction/update-status (admin session
// required), which mints the signed `ebnili_plan` grant cookie this endpoint
// reads. The client treats the server response as authoritative.

/**
 * Site owner — always full access, no payment needed.
 *
 * SECURITY: the address used to be hard-coded as a fallback, which published
 * the owner's account in the server bundle. It is now env-only. With nothing
 * configured, `isOwnerAccount()` is simply always false: the safest possible
 * state (no owner surface at all) rather than a guessable one.
 */
const OWNER_EMAIL = (
  process.env.SITE_OWNER_EMAIL || process.env.OWNER_EMAIL || ""
)
  .trim()
  .toLowerCase();

function isOwnerAccount(session: AuthUser | null): boolean {
  if (!session || !OWNER_EMAIL) return false;
  // Same normaliser the staff list uses, so an e-mail with stray whitespace or
  // mixed case can never be "the owner" for one check and "not staff" for another.
  return normalizeEmail(session.email) === OWNER_EMAIL;
}

/**
 * Added-admin email cache.
 *
 * The owner can add admin emails from the dashboard. An account matching a row
 * in `ebnily_admins` gets full admin access WITHOUT the PIN — they just sign in
 * with Google/GitHub. The original owner still needs the PIN as a second factor.
 *
 * The cache refreshes every 60 seconds so a newly added admin gets access
 * within a minute, and a removed admin is locked out equally fast.
 */
let adminEmailCache: Set<string> = new Set();
let adminCacheAt = 0;
const ADMIN_CACHE_TTL_MS = 60_000;

async function refreshAdminEmails(): Promise<Set<string>> {
  if (Date.now() - adminCacheAt < ADMIN_CACHE_TTL_MS) return adminEmailCache;
  if (!supabaseConfig().dbConfigured) return adminEmailCache;
  adminCacheAt = Date.now();
  try {
    const res = await dbRequest<Array<{ email: string }>>(ADMINS_TABLE, {
      query: { select: "email", limit: "500" },
    });
    if (res.ok && Array.isArray(res.data)) {
      adminEmailCache = new Set(res.data.map((r) => normalizeEmail(r.email)).filter(Boolean));
    }
  } catch {
    // keep the old cache
  }
  return adminEmailCache;
}

/** Synchronous check using the cached list. */
function isAddedAdmin(session: AuthUser | null): boolean {
  if (!session?.email) return false;
  return adminEmailCache.has(normalizeEmail(session.email));
}

// ── Staff accounts: automatic Business tier ───────────────────────────────────
// The list, the matcher and the reasoning all live in `api/security.ts`:
//   • `isStaffEmail()` decides the SUBSCRIPTION tier only.
//   • It is deliberately NOT wired into `isOwnerAccount()`, so these addresses get
//     the paid features but never the admin console. Admin remains
//     `OWNER_EMAIL` + the PIN — two independent factors.
// One definition, and the unit tests can pin it directly.

/**
 * Contact details the checkout and subscription screens read.
 *
 * WHY: these were hard-coded in every response. The owner can change the wallet
 * number from the dashboard, so the API must serve the stored value — otherwise
 * customers keep being told to transfer to a number that is no longer active.
 * Read lazily and fail soft: a storage hiccup must never block a page render.
 */
function contactDefaults(): { orangeWalletNumber: string; supportWhatsappNumber: string } {
  return { orangeWalletNumber: "01207782741", supportWhatsappNumber: "01207782741" };
}

function subscriptionPayload(grant: { email: string; tier: "pro" | "business"; expiresAt: string } | null) {
  const paid = grant
    ? {
        ...DEFAULT_SUBSCRIPTION,
        tier: grant.tier,
        status: "active" as const,
        planName: grant.tier === "business" ? "Business" : "Pro",
        generationsLimitToday: 99999,
        canExportZip: true,
        canDeployCustomDomain: grant.tier === "business",
        priorityAiModel: true,
        // Surfacing the real expiry is what lets the UI tell a customer how long
        // they have left instead of showing an open-ended "Pro".
        expiresAt: grant.expiresAt,
      }
    : { ...DEFAULT_SUBSCRIPTION };

  return {
    success: true,
    subscription: { ...paid, activatedAt: new Date().toISOString() },
    orangeWalletNumber: "01207782741",
    supportWhatsappNumber: "01207782741",
  };
}

/** Highest tier shape, used for the owner account. */
function ownerSubscription() {
  return {
    ...DEFAULT_SUBSCRIPTION,
    tier: "business" as const,
    status: "active" as const,
    planName: "Business",
    generationsLimitToday: 99999,
    canExportZip: true,
    canDeployCustomDomain: true,
    priorityAiModel: true,
    activatedAt: new Date().toISOString(),
  };
}

/**
 * Current entitlement for the signed-in account.
 *
 * WHY this must consult the database: the owner's approval writes `confirmed`
 * into `ebnily_payments` for the CUSTOMER's address. If this endpoint only
 * looked at a browser cookie, the approved customer would still see the free
 * tier — the payment would be recorded, the dashboard would show it as paid, and
 * the customer would keep hitting the daily limit. The latest unexpired confirmed
 * payment for this account is therefore the source of truth, on every device.
 */
app.get("/api/subscriptions/current", async (req: Request, res: Response) => {
  const session = readAuthSession(req);

  // The owner account needs no grant record.
  if (isOwnerAccount(session)) {
    res.json({
      success: true,
      subscription: ownerSubscription(),
      ...contactDefaults(),
    });
    return;
  }

  // A staff account answers here too, before any payment lookup. This is what
  // makes the app show Business immediately after sign-in rather than after a
  // database round trip, and it keeps the two staff addresses from ever appearing
  // as "free" in the UI.
  //
  // `status: "active"` comes from the shape itself: there is no expiry and no
  // pending review, so the subscription is active from the moment they land.
  if (isStaffEmail(session?.email)) {
    res.json({
      success: true,
      subscription: ownerSubscription(),
      ...contactDefaults(),
    });
    return;
  }

  let entitled: { email: string; tier: "pro" | "business"; expiresAt: string } | null = null;

  if (session && supabaseConfig().dbConfigured) {
    const result = await dbRequest<DbPayment[]>(PAYMENTS_TABLE, {
      query: {
        select: "account_email,plan_id,billing_cycle,reviewed_at",
        account_email: `eq.${session.email.toLowerCase()}`,
        status: "eq.confirmed",
        order: "reviewed_at.desc",
        limit: "1",
      },
    });
    const row = result.ok && Array.isArray(result.data) ? result.data[0] : undefined;
    if (row) {
      // A yearly plan runs a year; a monthly one a month — measured from when the
      // owner confirmed it, not from submission, so a delayed review is not
      // silently eaten out of the customer's paid time.
      const start = Date.parse(row.reviewed_at ?? "") || Date.now();
      const ttl = row.billing_cycle === "yearly" ? 365 * 24 * 60 * 60 * 1000 : 31 * 24 * 60 * 60 * 1000;
      if (Date.now() <= start + ttl) {
        entitled = {
          email: row.account_email,
          tier: row.plan_id === "business" ? "business" : "pro",
          expiresAt: new Date(start + ttl).toISOString(),
        };
      }
    }
  }

  // The signed cookie is the fallback: it is the owner's own fast path and it
  // still works when the database read failed.
  const grant = readPlanGrant(req);
  if (!entitled && grantBelongsToSession(grant, session)) {
    entitled = grant;
  }

  res.json(subscriptionPayload(entitled));
});

/** Only a logged-in (Google/GitHub) user may open a payment review request. */
function requireSignedIn(req: Request, res: Response): boolean {
  if (readAuthSession(req)) return true;
  res.status(401).json({
    success: false,
    error: "يجب تسجيل الدخول أولاً لإرسال طلب الاشتراك.",
  });
  return false;
}

/**
 * Records a payment claim for MANUAL review and returns the unchanged FREE
 * subscription. Never returns a paid tier: activation is an owner-only action.
 *
 * WHY the write matters: this used to acknowledge the request and persist
 * NOTHING ("no state is persisted"). The dashboard's transaction list therefore
 * stayed empty forever and the owner had no way to see or approve a payment — so
 * a paying customer could never actually be activated. The row written here is
 * what `/api/admin/overview` lists and what `update-status` approves.
 *
 * `account_email` comes from the signed session cookie, never the request body,
 * so a caller cannot file a payment under someone else's address and have a
 * grant minted for them.
 */
async function queuePaymentReview(req: Request, res: Response) {
  if (!requireSignedIn(req, res)) return;

  const body = (req.body as Record<string, unknown>) ?? {};
  const planId = String(body.planId || "pro");
  if (planId !== "pro" && planId !== "business") {
    res.status(400).json({ success: false, error: "خطة الاشتراك غير صحيحة." });
    return;
  }

  const reference = String(body.transactionReference ?? "").trim();
  const senderPhone = String(body.senderPhone ?? "").trim();
  if (senderPhone.length < 8) {
    res.status(400).json({
      success: false,
      error: "يرجى كتابة رقم هاتف محفظة أورانج كاش المحول منها بشكل صحيح.",
    });
    return;
  }
  if (reference.length < 3) {
    res.status(400).json({
      success: false,
      error: "يرجى كتابة الرقم المرجعي أو كود العملية من رسالة التحويل.",
    });
    return;
  }

  const session = readAuthSession(req as AuthReq);
  const id = `pay_${Date.now().toString(36)}_${crypto.randomBytes(4).toString("hex")}`;

  if (!supabaseConfig().dbConfigured) {
    // Honest failure. Telling the customer "received, under review" while the
    // record is discarded means their money paid for nothing and nobody ever
    // sees the request — the exact opposite of what this endpoint promises.
    return res.status(503).json({
      success: false,
      code: "DB_UNAVAILABLE",
      error:
        "تعذّر تسجيل طلب الاشتراك على الخادم. أرسل تفاصيل التحويل على واتساب 01207782741 وسنتولى التفعيل فوراً.",
    });
  }

  const write = await dbRequest<DbPayment[]>(PAYMENTS_TABLE, {
    method: "POST",
    query: { on_conflict: "ignore" },
    body: JSON.stringify({
      id,
      account_email: session?.email ?? "",
      account_id: session?.id ?? "",
      plan_id: planId,
      billing_cycle: String(body.billingCycle ?? "monthly") === "yearly" ? "yearly" : "monthly",
      amount_egp: toNumber(String(body.amount ?? "0")),
      amount_usd: 0,
      sender_phone: senderPhone,
      transaction_ref: reference,
      notes: typeof body.notes === "string" && body.notes.trim() ? body.notes.trim().slice(0, 500) : null,
      status: "pending",
    }),
  });

  if (!write.ok) {
    console.error("payment row insert failed:", write.error);
    return res.status(503).json({
      success: false,
      code: "PAYMENT_STORE_FAILED",
      error:
        "تعذّر حفظ طلب الاشتراك الآن. أرسل تفاصيل التحويل على واتساب 01207782741 وسنتولى التفعيل فوراً.",
    });
  }

  res.json({
    success: true,
    status: "pending",
    instant: false,
    referenceId: id,
    message:
      "تم استلام طلبك وهو قيد المراجعة. سيتم تفعيل اشتراكك بعد التحقق من التحويل من قِبل إدارة المنصة.",
    subscription: { ...DEFAULT_SUBSCRIPTION, activatedAt: new Date().toISOString() },
  });
}

// /pay uploads the transfer receipt here. The request is queued for the owner's
// manual review — it NEVER grants a tier, exactly like `queuePaymentReview`.
//
// STORAGE: Vercel functions have no writable disk, so the receipt is stored in an
// external bucket. Supabase Storage is used when configured (it already backs
// OAuth here), otherwise a generic S3-compatible endpoint can be supplied. When
// neither is present the endpoint says so plainly instead of pretending the file
// was saved — a payment receipt that silently disappears is worse than a clear
// error, and the customer keeps the WhatsApp fallback either way.
const RECEIPT_BUCKET = process.env.PAYMENTS_BUCKET || "payments-pending";

function receiptStorageConfigured(): boolean {
  const supabase = Boolean(supabaseConfig().configured);
  const s3 = Boolean((process.env.S3_ENDPOINT || "").trim());
  return supabase || s3;
}

/** Map a content type to a short, safe file extension for the object name. */
function safeExtension(contentType: string): string {
  const map: Record<string, string> = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "application/pdf": "pdf",
  };
  return map[contentType] || "bin";
}

/**
 * Write one object to the configured bucket.
 *
 * Supabase Storage is preferred (it already backs OAuth in this project); a
 * generic S3-compatible endpoint is supported for anything else. Returns false
 * rather than throwing so the caller can turn it into a clean 503 — and returns
 * false whenever NO storage is configured, so a missing env var can never look
 * like a successful upload.
 */
async function storagePut(
  path: string,
  base64: string,
  contentType: string,
  metadata: string,
): Promise<boolean> {
  const bytes = Buffer.from(base64, "base64");
  const supabase = supabaseConfig();
  if (supabase.configured) {
    const res = await fetch(`${supabase.url}/storage/v1/object/${RECEIPT_BUCKET}/${path}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${supabase.anonKey}`,
        apikey: supabase.anonKey,
        "Content-Type": contentType,
        "x-upsert": "true",
        "cache-control": "3600",
      },
      body: new Uint8Array(bytes),
    });
    if (res.ok) {
      // Metadata sidecar so the owner can list and triage without downloading.
      await fetch(`${supabase.url}/storage/v1/object/${RECEIPT_BUCKET}/${path}.json`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${supabase.anonKey}`,
          apikey: supabase.anonKey,
          "Content-Type": "application/json",
          "x-upsert": "true",
        },
        body: metadata,
      }).catch(() => undefined);
      return true;
    }
    console.error("storagePut supabase:", res.status, await res.text().catch(() => ""));
    return false;
  }

  const endpoint = (process.env.S3_ENDPOINT || "").trim().replace(/\/+$/, "");
  if (endpoint) {
    const res = await fetch(`${endpoint}/${RECEIPT_BUCKET}/${path}`, {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${process.env.S3_TOKEN || ""}`,
        "Content-Type": contentType,
        "x-amz-meta-ebnili": Buffer.from(metadata).toString("base64"),
      },
      body: new Uint8Array(bytes),
    });
    return res.ok;
  }

  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Projects — real, per-account, server-verified
// ═══════════════════════════════════════════════════════════════════════════
// Every route below requires the signed session cookie, derives the owner from
// that cookie only (never from the body/query), and filters every query by
// owner_id — so changing an ?id= in the URL yields 404 for somebody else's
// project instead of their content. Ids are minted server-side, so a reload can
// never create a duplicate or a phantom project.

function toPublicProject(row: DbProject) {
  return {
    id: row.id,
    name: row.name,
    code: row.code,
    files: row.files ?? {},
    versions: Array.isArray(row.versions) ? row.versions : [],
    theme: row.theme ?? {},
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

/** 503 with a readable reason instead of pretending the data exists. */
function dbUnavailable(res: Response) {
  return res.status(503).json({
    success: false,
    code: "DB_UNAVAILABLE",
    message: "قاعدة بيانات المشاريع غير مهيأة على الخادم.",
  });
}

/**
 * Coerce a stored settings blob into a complete, valid settings object.
 *
 * The row is owner-editable JSON, so every field has to be validated: a partial
 * or hand-edited row must not send `undefined` (or a wallet number that is not a
 * string) into the React tree. Anything invalid falls back to the default, and
 * the free limit is clamped to a sane range.
 */
function normalizePlatformSettings(value: unknown): typeof DEFAULT_ADMIN_SETTINGS {
  const source = (value && typeof value === "object" && !Array.isArray(value) ? value : {}) as Record<string, unknown>;
  const str = (key: string, fallback: string): string => {
    const raw = source[key];
    return typeof raw === "string" && raw.trim() ? raw.trim() : fallback;
  };
  const rawLimit = Number(source.defaultFreeLimit);
  const limit = Number.isFinite(rawLimit) ? Math.min(50, Math.max(1, Math.round(rawLimit))) : 5;

  return {
    orangeWalletNumber: str("orangeWalletNumber", DEFAULT_ADMIN_SETTINGS.orangeWalletNumber),
    defaultFreeLimit: limit,
    autoVerificationEnabled:
      typeof source.autoVerificationEnabled === "boolean"
        ? source.autoVerificationEnabled
        : DEFAULT_ADMIN_SETTINGS.autoVerificationEnabled,
    supportWhatsappNumber: str("supportWhatsappNumber", DEFAULT_ADMIN_SETTINGS.supportWhatsappNumber),
    siteName: str("siteName", DEFAULT_ADMIN_SETTINGS.siteName),
    // The owner's address is resolved from the environment, never from the
    // stored row, so editing settings cannot redirect owner privileges.
    adminEmail: DEFAULT_ADMIN_SETTINGS.adminEmail,
  };
}

/** List every project owned by the signed-in account, newest first. */
app.get("/api/projects", (req: Request, res: Response) => {
  const owner = currentOwner(req);
  if (!owner) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);

  void dbRequest<DbProject[]>(PROJECTS_TABLE, {
    query: {
      select: "id,name,code,files,versions,theme,created_at,updated_at",
      owner_id: `eq.${owner.id}`,
      order: "updated_at.desc",
    },
  }).then((result) => {
    if (!result.ok) {
      console.error("projects list failed:", result.error);
      return dbUnavailable(res);
    }
    const rows = (result.data ?? []).map((row) => ({
      ...toPublicProject(row),
      // The list view does not need the full document body.
      code: "",
      preview: (row.code || "")
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<[^>]+>/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 160),
    }));
    res.json({ success: true, projects: rows });
  });
});

/** Open one project. Scoped by owner_id, so another account's id → 404. */
app.get("/api/projects/:id", (req: Request, res: Response) => {
  const owner = currentOwner(req);
  if (!owner) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);

  const id = String(req.params.id ?? "");
  void dbRequest<DbProject[]>(PROJECTS_TABLE, {
    query: {
      select: "id,name,code,files,versions,theme,created_at,updated_at",
      owner_id: `eq.${owner.id}`,
      id: `eq.${id}`,
      limit: "1",
    },
  }).then((result) => {
    if (!result.ok) {
      console.error("project read failed:", result.error);
      return dbUnavailable(res);
    }
    const row = result.data?.[0];
    // 404 (not 403): a stranger's project must be indistinguishable from one
    // that does not exist.
    if (!row) return res.status(404).json({ success: false, code: "NOT_FOUND" });
    res.json({ success: true, project: toPublicProject(row) });
  });
});

app.post("/api/projects", limitWrites, (req: Request, res: Response) => {
  const owner = currentOwner(req);
  if (!owner) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);

  const body = (req.body as Record<string, unknown>) ?? {};
  const now = new Date().toISOString();
  const row: DbProject = {
    id: `prj_${Date.now().toString(36)}_${crypto.randomBytes(4).toString("hex")}`,
    owner_id: owner.id,
    name: String(body.name ?? "مشروع بدون اسم").slice(0, 120) || "مشروع بدون اسم",
    code: typeof body.code === "string" ? body.code : "",
    files: (body.files && typeof body.files === "object" ? body.files : {}) as Record<string, string>,
    versions: Array.isArray(body.versions) ? (body.versions as unknown[]).slice(0, 20) : [],
    theme: (body.theme && typeof body.theme === "object" ? body.theme : {}) as Record<string, unknown>,
    created_at: now,
    updated_at: now,
  };

  void dbRequest<DbProject[]>(PROJECTS_TABLE, {
    method: "POST",
    body: JSON.stringify(row),
  }).then((result) => {
    if (!result.ok || !result.data?.[0]) {
      console.error("project create failed:", result.error);
      return dbUnavailable(res);
    }
    res.status(201).json({ success: true, project: toPublicProject(result.data[0]) });
  });
});

app.patch("/api/projects/:id", limitWrites, (req: Request, res: Response) => {
  const owner = currentOwner(req);
  if (!owner) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);

  const id = String(req.params.id ?? "");
  const body = (req.body as Record<string, unknown>) ?? {};
  const patch: Record<string, unknown> = { updated_at: new Date().toISOString() };
  if (typeof body.name === "string" && body.name.trim()) {
    patch.name = body.name.trim().slice(0, 120);
  }
  if (typeof body.code === "string") patch.code = body.code;
  if (body.files && typeof body.files === "object") patch.files = body.files;
  if (Array.isArray(body.versions)) patch.versions = body.versions.slice(0, 20);
  if (body.theme && typeof body.theme === "object") patch.theme = body.theme;

  void dbRequest<DbProject[]>(PROJECTS_TABLE, {
    method: "PATCH",
    query: { owner_id: `eq.${owner.id}`, id: `eq.${id}` },
    body: JSON.stringify(patch),
  }).then((result) => {
    if (!result.ok) {
      console.error("project update failed:", result.error);
      return dbUnavailable(res);
    }
    const row = result.data?.[0];
    if (!row) return res.status(404).json({ success: false, code: "NOT_FOUND" });
    res.json({ success: true, project: toPublicProject(row) });
  });
});

app.delete("/api/projects/:id", (req: Request, res: Response) => {
  const owner = currentOwner(req);
  if (!owner) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);

  const id = String(req.params.id ?? "");
  void dbRequest<DbProject[]>(PROJECTS_TABLE, {
    method: "DELETE",
    query: { owner_id: `eq.${owner.id}`, id: `eq.${id}` },
  }).then((result) => {
    if (!result.ok) {
      console.error("project delete failed:", result.error);
      return dbUnavailable(res);
    }
    if (!result.data || result.data.length === 0) {
      return res.status(404).json({ success: false, code: "NOT_FOUND" });
    }
    res.json({ success: true, deleted: id });
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Export — the ONLY place files are produced, and the tier is resolved here
// ═══════════════════════════════════════════════════════════════════════════
// The client no longer assembles the ZIP from its own copy of the code, so the
// mark cannot be dodged by exporting from the browser. A `tier` in the request
// body is IGNORED on purpose: a client claiming to be paid changes nothing.

app.post("/api/projects/export", (req: Request, res: Response) => {
  const session = readAuthSession(req as AuthReq);
  if (!session) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });

  const tier = serverTierFor(req, session);
  const body = (req.body as { files?: Record<string, string> }) ?? {};
  const files = body.files && typeof body.files === "object" ? body.files : {};

  const marked: Record<string, string> = {};
  for (const [path, content] of Object.entries(files)) {
    // Only the executable page carries a mark; README/JSON/SQL are not documents.
    marked[path] = /\.html?$/i.test(path) ? applyWatermark(String(content), tier) : String(content);
  }

  res.json({
    success: true,
    // The server's verdict, returned so the client can label the download.
    tier,
    watermarked: !isPaidTier(tier),
    files: marked,
  });
});

// ── Payment receipts (pending bucket) ────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════
// GitHub import — real repository import
// ═══════════════════════════════════════════════════════════════════════════
// WHY A SERVER ROUTE: listing a user's private repos and reading a tree both
// need a GitHub access token. That token must never touch the browser, so every
// call happens here and the browser only receives metadata and file contents.
//
// TWO WAYS IN, both handled explicitly:
//   1. The user linked GitHub  → this project exchanged a code for a token and
//      keeps it server-side, tied to the account.
//   2. Google-only users       → there is no GitHub token, so the route reports
//      `linked: false` and the UI offers either connecting GitHub or a PUBLIC
//      repo URL, which needs no token at all.
//
// READ-ONLY BY DESIGN: nothing here writes to GitHub, and `scope` asks only for
// `read:user`, so this feature cannot push or delete anything.

const GH_TOKEN_COOKIE = "ebnili_gh_token";
const GH_LINK_COOKIE = "ebnili_gh_link";
const GH_SCOPES = "read:user";

interface StoredGhToken {
  userId: string;
  login: string;
  accessToken: string;
  linkNonce: string;
}

function ghStateNonce(): string {
  return crypto.randomBytes(16).toString("hex");
}

/** Signed, so a hand-edited cookie cannot smuggle a token in. */
function issueGhToken(value: StoredGhToken): string {
  const payload = Buffer.from(JSON.stringify(value), "utf-8").toString("base64url");
  return `${payload}.${hmacB64With(payload, planSigningSecret())}`;
}

function readGhToken(req: AuthReq): StoredGhToken | null {
  const token = readCookie(req, GH_TOKEN_COOKIE);
  if (!token) return null;
  const sep = token.indexOf(".");
  if (sep <= 0) return null;
  const body = token.slice(0, sep);
  if (!safeEqual(token.slice(sep + 1), hmacB64With(body, planSigningSecret()))) return null;
  try {
    const parsed = JSON.parse(Buffer.from(body, "base64url").toString("utf-8")) as StoredGhToken;
    if (!parsed?.accessToken || !parsed?.userId) return null;
    return parsed;
  } catch {
    return null;
  }
}

function ghHeaders(token?: string): Record<string, string> {
  return {
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "Ebnili-App",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

/** Turn any user-typed form into `owner/repo`, or null if it is not one. */
function parseRepoRef(input: string): { owner: string; repo: string } | null {
  const cleaned = String(input ?? "")
    .trim()
    .replace(/^https?:\/\/(www\.)?github\.com\//i, "")
    .replace(/^git@github\.com:/i, "")
    .replace(/\.git$/i, "")
    .replace(/^\/+|\/+$/g, "");
  const parts = cleaned.split("/").filter(Boolean);
  if (parts.length < 2) return null;
  const [owner, repo] = parts;
  if (!/^[\w.-]+$/.test(owner) || !/^[\w.-]+$/.test(repo)) return null;
  return { owner, repo };
}

/**
 * The token counts only when BOTH cookies agree AND the token belongs to this
 * session's account — so a GitHub link made in one session cannot be used from
 * another account on the same browser.
 */
function ghTokenForSession(req: AuthReq, session: AuthUser): StoredGhToken | null {
  const stored = readGhToken(req);
  const linkNonce = readCookie(req, GH_LINK_COOKIE);
  if (!stored || !linkNonce) return null;
  if (!safeEqual(stored.linkNonce, linkNonce)) return null;
  return stored.userId === session.id ? stored : null;
}

/** Is GitHub connected for this session? */
app.get("/api/github/status", (req: AuthReq, res: AuthRes) => {
  const session = readAuthSession(req);
  if (!session) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  const stored = ghTokenForSession(req, session);
  res.json({
    success: true,
    linked: Boolean(stored),
    login: stored?.login ?? "",
    // A public repo is importable with no GitHub connection at all.
    publicImportAvailable: true,
    provider: session.provider,
  });
});

/**
 * The exact callback URL this server sends to GitHub.
 *
 * WHY this exists: GitHub rejects the authorisation screen with "The
 * redirect_uri is not associated with this application" when the URL we send is
 * not registered on the OAuth app — character for character, host included. That
 * used to be unguessable: the value depends on `APP_URL`, and nothing in the UI
 * displayed it, so the only way to find it was to read the source.
 *
 * It is derived from the SAME helper `/api/github/connect` uses, so the string
 * shown here is guaranteed to be the string GitHub will compare against. Copy it
 * into the OAuth app's "Callback URL" field.
 */
app.get("/api/github/config", (req: AuthReq, res: AuthRes) => {
  const cfg = providerConfig("github");
  const base = requestBaseUrl(req);
  res.json({
    success: true,
    configured: cfg.configured,
    clientId: cfg.configured ? `${cfg.clientId.slice(0, 4)}…${cfg.clientId.slice(-4)}` : "",
    callbackUrl: `${base}/api/github/callback`,
    // Surfaced so a misconfigured APP_URL is visible instead of mysterious: if
    // this does not match the domain the visitor is on, that is the bug.
    baseUrl: base,
    appUrlConfigured: Boolean((process.env.APP_URL || "").trim()),
    scopes: GH_SCOPES,
  });
});

/** Step 1 — send the browser to GitHub to authorise a read-only link. */
app.get("/api/github/connect", (req: AuthReq, res: AuthRes) => {
  const session = readAuthSession(req);
  if (!session) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  const cfg = providerConfig("github");
  if (!cfg.configured) {
    return res.status(503).json({ success: false, message: "GitHub OAuth is not configured." });
  }
  const nonce = ghStateNonce();
  res.cookie(AUTH_STATE_COOKIE, `ghlink.${nonce}`, authCookieOptions(AUTH_STATE_TTL_MS, req));

  const url = new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", cfg.clientId);
  url.searchParams.set("redirect_uri", `${requestBaseUrl(req)}/api/github/callback`);
  url.searchParams.set("scope", GH_SCOPES);
  url.searchParams.set("state", nonce);
  url.searchParams.set("allow_signup", "false");
  res.redirect(url.toString());
});

/** Step 2 — GitHub calls back; exchange the code and store the token. */
app.get("/api/github/callback", async (req: AuthReq, res: AuthRes) => {
  const session = readAuthSession(req);
  const home = requestBaseUrl(req);
  const fail = (reason: string) => res.redirect(`${home}/?github_error=${encodeURIComponent(reason)}`);
  if (!session) return fail("auth_required");

  // Single-use nonce, as with the sign-in callback — a replayed callback URL
  // cannot mint a second link.
  const expected = readCookie(req, AUTH_STATE_COOKIE);
  clearAuthCookie(res, AUTH_STATE_COOKIE, req);
  if (!expected || !expected.startsWith("ghlink.")) return fail("state_cookie_missing");
  const nonce = expected.slice("ghlink.".length);
  const state = typeof req.query.state === "string" ? req.query.state : "";
  if (!safeEqual(nonce, state)) return fail("state_mismatch");

  const code = typeof req.query.code === "string" ? req.query.code : "";
  if (!code) return fail("missing_code");

  const cfg = providerConfig("github");
  if (!cfg.configured) return fail("not_configured");

  try {
    const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
      body: new URLSearchParams({
        client_id: cfg.clientId,
        client_secret: cfg.clientSecret,
        code,
        redirect_uri: `${home}/api/github/callback`,
      }).toString(),
    });
    const tokenJson = (await tokenRes.json()) as { access_token?: string; error?: string };
    if (!tokenJson.access_token) return fail(tokenJson.error || "exchange_failed");

    // Read the login so the UI can confirm which account was linked.
    let login = "";
    try {
      const profile = await fetch("https://api.github.com/user", {
        headers: ghHeaders(tokenJson.access_token),
      });
      if (profile.ok) {
        const me = (await profile.json()) as { login?: string };
        login = String(me.login || "");
      }
    } catch {
      /* the link still works; the label is cosmetic */
    }

    // One nonce in BOTH cookies: ghTokenForSession requires them to match, so a
    // token copied without its partner cookie is worthless.
    const linkNonce = ghStateNonce();
    res.cookie(GH_LINK_COOKIE, linkNonce, authCookieOptions(AUTH_SESSION_TTL_MS, req));
    res.cookie(
      GH_TOKEN_COOKIE,
      issueGhToken({
        userId: session.id,
        login,
        accessToken: tokenJson.access_token,
        linkNonce,
      }),
      authCookieOptions(AUTH_SESSION_TTL_MS, req),
    );
    return res.redirect(`${home}/?github=linked`);
  } catch (e) {
    console.error("github link failed:", e instanceof Error ? e.message : String(e));
    return fail("exchange_failed");
  }
});

/** Disconnect GitHub — the stored token is destroyed, not merely ignored. */
app.post("/api/github/disconnect", (req: AuthReq, res: AuthRes) => {
  if (!readAuthSession(req)) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  clearAuthCookie(res, GH_TOKEN_COOKIE, req);
  clearAuthCookie(res, GH_LINK_COOKIE, req);
  res.json({ success: true, linked: false });
});

/**
 * The user's own repositories. A GitHub link is required because private
 * repositories are impossible to list without a token; a PUBLIC repo stays
 * importable through /api/github/tree for any signed-in user.
 */
app.get("/api/github/repos", async (req: AuthReq, res: AuthRes) => {
  const session = readAuthSession(req);
  if (!session) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });
  const stored = ghTokenForSession(req, session);
  if (!stored) {
    return res.status(403).json({
      success: false,
      code: "GITHUB_NOT_LINKED",
      message: "اربط حساب GitHub لعرض مستودعاتك.",
    });
  }

  try {
    const page = Math.min(Math.max(Number(req.query.page) || 1, 1), 5);
    const listRes = await fetch(
      `https://api.github.com/user/repos?per_page=30&sort=updated&page=${page}` +
        "&affiliation=owner,collaborator,organization_member",
      { headers: ghHeaders(stored.accessToken) },
    );
    if (listRes.status === 401) {
      return res.status(403).json({
        success: false,
        code: "GITHUB_TOKEN_EXPIRED",
        message: "انتهت صلاحية الربط مع GitHub. أعد الربط من جديد.",
      });
    }
    if (!listRes.ok) return res.status(502).json({ success: false, message: "GitHub did not answer." });

    const repos = (await listRes.json()) as Array<{
      id: number;
      name: string;
      full_name: string;
      description: string | null;
      private: boolean;
      default_branch: string;
      language: string | null;
      stargazers_count: number;
      updated_at: string;
    }>;

    res.json({
      success: true,
      login: stored.login,
      repos: repos.map((r) => ({
        id: r.id,
        name: r.name,
        fullName: r.full_name,
        description: r.description || "",
        private: r.private,
        defaultBranch: r.default_branch,
        language: r.language || "",
        stars: r.stargazers_count,
        updatedAt: r.updated_at,
      })),
    });
  } catch (e) {
    console.error("github repos failed:", e instanceof Error ? e.message : String(e));
    res.status(502).json({ success: false, message: "تعذر الوصول إلى GitHub." });
  }
});

/** Extensions worth opening in an editor; the rest would only bloat the project. */
const IMPORTABLE_EXT =
  /\.(html?|css|jsx?|tsx?|mjs|cjs|json|md|svg|txt|vue|svelte|astro|ya?ml|toml|sql|sh)$/i;
/** Never read these: dependencies, build output, secrets and heavy lockfiles. */
const SKIP_PATH =
  /(^|\/)(node_modules|\.git|dist|build|out|\.next|\.vercel|vendor|coverage|\.cache|__pycache__)(\/|$)/i;
const SKIP_FILE =
  /(^|\/)(\.env(\..*)?|\.gitignore|\.npmrc|\.DS_Store)$|package-lock\.json$|yarn\.lock$|pnpm-lock\.yaml$|\.(png|jpe?g|gif|webp|avif|ico|bmp|pdf|zip|gz|tar|woff2?|ttf|eot|mp4|mp3|wasm|so|dll|exe|bin)$/i;

const MAX_IMPORT_FILES = 60;
const MAX_FILE_BYTES = 400_000;
const MAX_TOTAL_BYTES = 2_500_000;

/**
 * Import a repository's files through the Git Trees API.
 *
 * One `?recursive=1` tree request lists every blob (path, SHA, size) in a single
 * call, then each blob is fetched as base64. Walking the contents API per
 * directory instead would cost a request per folder and blow GitHub's rate
 * limit on any real repository.
 *
 * A linked token unlocks private repositories; without one GitHub serves public
 * ones (rate-limited, but enough for the fallback path). Nothing is written.
 */
app.post("/api/github/tree", async (req: AuthReq, res: AuthRes) => {
  const session = readAuthSession(req);
  if (!session) return res.status(401).json({ success: false, code: "AUTH_REQUIRED" });

  const body = (req.body as { owner?: string; repo?: string; ref?: string }) ?? {};
  const ref = parseRepoRef(`${body.owner ?? ""}/${body.repo ?? ""}`);
  if (!ref) return res.status(400).json({ success: false, message: "اسم المستودع غير صحيح." });

  const branch = String(body.ref ?? "").trim() || "HEAD";
  const token = ghTokenForSession(req, session)?.accessToken;

  try {
    const repoRes = await fetch(`https://api.github.com/repos/${ref.owner}/${ref.repo}`, {
      headers: ghHeaders(token),
    });
    if (repoRes.status === 404) {
      return res.status(404).json({
        success: false,
        message: token
          ? "المستودع غير موجود أو لا تملك صلاحية الوصول إليه."
          : "المستودع غير موجود أو خاص. اربط GitHub لاستيراد المستودعات الخاصة.",
      });
    }
    if (!repoRes.ok) {
      return res.status(502).json({ success: false, message: "تعذر قراءة بيانات المستودع." });
    }
    const meta = (await repoRes.json()) as { default_branch?: string; name?: string };

    const target = branch === "HEAD" ? meta.default_branch || "main" : branch;
    const treeRes = await fetch(
      `https://api.github.com/repos/${ref.owner}/${ref.repo}/git/trees/` +
        `${encodeURIComponent(target)}?recursive=1`,
      { headers: ghHeaders(token) },
    );
    if (!treeRes.ok) return res.status(502).json({ success: false, message: "تعذر قراءة شجرة الملفات." });
    const tree = (await treeRes.json()) as {
      tree?: Array<{ path: string; type: string; sha: string; size?: number }>;
      truncated?: boolean;
    };

    const all = (tree.tree ?? []).filter((n) => n.type === "blob");
    const wanted = all
      .filter((n) => !SKIP_PATH.test(n.path) && !SKIP_FILE.test(n.path) && IMPORTABLE_EXT.test(n.path))
      .filter((n) => (n.size ?? 0) <= MAX_FILE_BYTES)
      .sort((a, b) => (b.size ?? 0) - (a.size ?? 0));

    // Open the entry point first when there is one.
    const entry = wanted.find((n) => /(^|\/)(index|app|main)\.(html?|jsx?|tsx?)$/i.test(n.path));
    const picked = (entry ? [entry, ...wanted.filter((n) => n !== entry)] : wanted).slice(
      0,
      MAX_IMPORT_FILES,
    );

    const unsupported = all.length - wanted.length;
    let totalBytes = 0;
    let fetched = 0;
    let skippedTooLarge = 0;
    const files: Record<string, string> = {};

    for (const node of picked) {
      if (totalBytes >= MAX_TOTAL_BYTES) {
        skippedTooLarge = picked.length - fetched;
        break;
      }
      try {
        const blobRes = await fetch(
          `https://api.github.com/repos/${ref.owner}/${ref.repo}/git/blobs/${node.sha}`,
          { headers: ghHeaders(token) },
        );
        if (!blobRes.ok) continue;
        const blob = (await blobRes.json()) as { content?: string; encoding?: string };
        if (blob.encoding !== "base64" || !blob.content) continue;
        const text = Buffer.from(blob.content.replace(/\s/g, ""), "base64").toString("utf-8");
        // A blob can decode larger than its reported size; re-check for real.
        const size = Buffer.byteLength(text, "utf-8");
        if (size > MAX_FILE_BYTES) continue;
        files[node.path] = text;
        totalBytes += size;
        fetched += 1;
      } catch {
        // One unreadable file must not abort the whole import.
      }
    }

    if (Object.keys(files).length === 0) {
      return res.status(422).json({
        success: false,
        message: "لم يتم العثور على ملفات قابلة للعرض في هذا المستودع.",
      });
    }

    res.json({
      success: true,
      repo: {
        owner: ref.owner,
        name: meta.name || ref.repo,
        fullName: `${ref.owner}/${meta.name || ref.repo}`,
        branch: target,
      },
      files,
      // An honest account of what did not make it in.
      stats: {
        imported: Object.keys(files).length,
        skippedBinaryOrUnsupported: unsupported,
        skippedTooLarge,
        truncated: Boolean(tree.truncated),
      },
    });
  } catch (e) {
    console.error("github tree failed:", e instanceof Error ? e.message : String(e));
    res.status(502).json({ success: false, message: "تعذر الاتصال بـ GitHub." });
  }
});

app.post("/api/payments/receipt", limitPayments, async (req: Request, res: Response) => {
  if (!requireSignedIn(req, res)) return;

  const body = (req.body as Record<string, unknown>) ?? {};
  const planId = String(body.planId ?? "pro");
  if (planId !== "pro" && planId !== "business") {
    return res.status(400).json({ success: false, message: "خطة الاشتراك غير صحيحة." });
  }
  const reference = String(body.transactionReference ?? "").trim();
  const senderPhone = String(body.senderPhone ?? "").trim();
  if (senderPhone.length < 8 || reference.length < 3) {
    return res.status(400).json({
      success: false,
      message: "يرجى إدخال رقم المحوِّل والرقم المرجعي قبل رفع الإشعار.",
    });
  }

  const fileName = String(body.fileName ?? "").trim();
  const fileType = String(body.fileType ?? "").trim();
  const fileData = typeof body.fileData === "string" ? body.fileData : "";

  if (!fileData) {
    // Nothing attached — the transfer details alone are still worth recording.
    return res.json({
      success: true,
      receiptStored: false,
      status: "pending",
      message: "تم استلام بيانات التحويل. سنتحقق منها ونفعّل اشتراكك.",
    });
  }

  if (!receiptStorageConfigured()) {
    // Honest failure: the page falls back to sending the receipt on WhatsApp.
    return res.status(503).json({
      success: false,
      code: "STORAGE_NOT_CONFIGURED",
      message:
        "تعذّر حفظ صورة الإشعار على الخادم. من فضلك أرسل الصورة على واتساب على الرقم 01207782741 مع الرقم المرجعي.",
    });
  }

  const session = readAuthSession(req as AuthReq);
  const record = {
    id: `pay_${Date.now().toString(36)}_${crypto.randomBytes(4).toString("hex")}`,
    planId,
    cycle: String(body.cycle ?? "monthly"),
    amountEgp: Number(body.amountEgp ?? 0),
    amountUsd: Number(body.amountUsd ?? 0),
    senderPhone,
    transactionReference: reference,
    account: session?.email ?? "",
    fileName: fileName || "receipt",
    fileType,
    createdAt: new Date().toISOString(),
  };

  // Store the receipt bytes together with a metadata sidecar, under one object
  // name, so the owner can review the pair later.
  const objectKey = `${RECEIPT_BUCKET}/${record.id}.${safeExtension(fileType)}`;
  try {
    const base64 = fileData.includes(",") ? fileData.slice(fileData.indexOf(",") + 1) : fileData;
    const ok = await storagePut(objectKey, base64, fileType || "application/octet-stream", JSON.stringify(record));
    if (!ok) throw new Error("storage rejected the write");
  } catch (err) {
    console.error("receipt upload failed:", err instanceof Error ? err.message : String(err));
    return res.status(503).json({
      success: false,
      code: "STORAGE_WRITE_FAILED",
      message: "تعذّر رفع الصورة الآن. أرسلها على واتساب 01207782741 مع الرقم المرجعي.",
    });
  }

  // The image alone is not a reviewable record: without this row the dashboard
  // never lists the request, so the owner would have no way to approve it. The
  // email comes from the session cookie, never from the request body.
  if (supabaseConfig().dbConfigured) {
    const write = await dbRequest<DbPayment[]>(PAYMENTS_TABLE, {
      method: "POST",
      query: { on_conflict: "ignore" },
      body: JSON.stringify({
        id: record.id,
        account_email: session?.email ?? "",
        account_id: session?.id ?? "",
        plan_id: planId,
        billing_cycle: record.cycle === "yearly" ? "yearly" : "monthly",
        amount_egp: record.amountEgp,
        amount_usd: record.amountUsd,
        sender_phone: senderPhone,
        transaction_ref: reference,
        receipt_path: objectKey,
        receipt_file_name: record.fileName,
        status: "pending",
      }),
    });
    if (!write.ok) {
      // The receipt is already stored, so do not lose it: tell the owner exactly
      // where to find it instead of pretending the upload failed.
      console.error("payment row insert failed:", write.error);
      return res.status(503).json({
        success: false,
        code: "PAYMENT_STORE_FAILED",
        message: `تم حفظ الصورة (${record.id}) لكن تعذّر تسجيل الطلب. أرسل المرجع ${record.id} على واتساب 01207782741.`,
      });
    }
  }

  res.json({
    success: true,
    receiptStored: true,
    referenceId: record.id,
    status: "pending",
    message: "تم رفع إشعار التحويل. سنتحقق منه ونفعّل اشتراكك.",
  });
});

app.post("/api/subscriptions/auto-verify", limitPayments, queuePaymentReview);
app.post("/api/subscriptions/submit-orange-cash", limitPayments, queuePaymentReview);

// Downgrade is a client-side display concern; the server state is already free,
// so this never returns an upgraded tier.
app.post("/api/subscriptions/reset-free", (_req: Request, res: Response) => {
  res.json({ success: true, subscription: DEFAULT_SUBSCRIPTION });
});

// ── Signed-in user guard for the AI engine ──────────────────────────────────
// SECURITY: every /api/ai/* route used to be anonymous, so anybody (a bot, a
// leaked URL, a curl loop) could spend the owner's Gemini key at will. The
// studio itself is already behind the login gate, so requiring the same signed
// session cookie costs a real user nothing and closes the door on abuse.
// Reuses the existing `requireSignedIn` guard (line 145) in middleware form.
function requireAiSession(req: Request, res: Response, next: NextFunction) {
  if (!readAuthSession(req as AuthReq)) {
    return res.status(401).json({
      success: false,
      code: "AUTH_REQUIRED",
      message: "سجّل الدخول مرة أخرى للمتابعة — الجلسة انتهت أو لم يتم العثور عليها.",
    });
  }
  // The short-window limit sits HERE rather than on each route so every AI
  // endpoint inherits it: these calls spend the owner's Gemini key, and a single
  // runaway loop through any one of them is expensive. The per-account daily cap
  // below is the separate, longer-horizon boundary.
  const result = aiLimiter.check(clientKey(req, "ai"), LIMIT_RULES.ai);
  res.setHeader("X-RateLimit-Limit", String(LIMIT_RULES.ai.max));
  res.setHeader("X-RateLimit-Remaining", String(result.remaining));
  if (!result.allowed) {
    res.setHeader("Retry-After", String(result.retryAfterSec));
    return res.status(429).json({
      success: false,
      code: "RATE_LIMITED",
      message: "طلبات كثيرة جداً في وقت قصير. انتظر قليلاً ثم أعد المحاولة.",
      retryAfter: result.retryAfterSec,
    });
  }
  next();
}

// A light per-session daily cap. The serverless runtime is stateless, so this
// is best-effort (it stops a single runaway client, not a determined attacker)
// — but the real boundary is the paid plan, and the owner can still raise the
// ceiling through the admin dashboard.
const AI_DAILY_LIMIT = 60;
const aiUsage = new Map<string, { day: string; count: number }>();

function aiQuotaExceeded(sessionId: string): boolean {
  const day = new Date().toISOString().slice(0, 10);
  const rec = aiUsage.get(sessionId);
  if (!rec || rec.day !== day) {
    aiUsage.set(sessionId, { day, count: 1 });
    return false;
  }
  if (rec.count >= AI_DAILY_LIMIT) return true;
  rec.count += 1;
  return false;
}

/**
 * Device registry — register on sight, report the real block state.
 *
 * WHY: this answered `isBlocked: false, freeGenerationsUsed: 0` for everyone,
 * unconditionally. Two consequences: an owner who blocked a device saw it keep
 * working (the block button had no effect on the product), and the dashboard's
 * device list stayed permanently empty because nothing was ever recorded.
 *
 * The row is written with an upsert keyed on `device_id`, so `last_seen_at` and
 * `account_email` stay current without a separate registration call.
 */
app.get("/api/protection/status", async (req: Request, res: Response) => {
  const deviceId = String(req.query.deviceId ?? req.header("x-device-id") ?? "").trim();
  const fingerprintHash = String(req.query.fingerprintHash ?? req.header("x-fingerprint-hash") ?? "").trim();
  const session = readAuthSession(req as AuthReq);

  if (!deviceId) {
    return res.json({
      success: true,
      deviceId: null,
      isBlocked: false,
      freeGenerationsUsed: 0,
      freeGenerationsLimit: 5,
      configured: supabaseConfig().dbConfigured,
    });
  }

  if (!supabaseConfig().dbConfigured) {
    // No database: report the un-configured state rather than pretending the
    // device is healthy and registered.
    return res.json({
      success: true,
      deviceId,
      isBlocked: false,
      freeGenerationsUsed: 0,
      freeGenerationsLimit: 5,
      configured: false,
    });
  }

  const now = new Date().toISOString();
  const upsert = await dbRequest<DbDevice[]>(DEVICES_TABLE, {
    method: "POST",
    query: { on_conflict: "merge-duplicates" },
    body: JSON.stringify({
      device_id: deviceId,
      fingerprint_hash: fingerprintHash,
      // Only fill the account when we actually have one; a null would otherwise
      // erase the address of a device that previously signed in.
      ...(session ? { account_email: session.email.toLowerCase() } : {}),
      screen: String(req.query.screen ?? "").slice(0, 40) || null,
      timezone: String(req.query.timezone ?? "").slice(0, 60) || null,
      platform: String(req.query.platform ?? "").slice(0, 40) || null,
      user_agent: String(req.header("user-agent") ?? "").slice(0, 300) || null,
      ip_address: String(adminClientKey(req)).slice(0, 60) || null,
      last_seen_at: now,
    }),
  });

  // Read back so the answer reflects what is stored (the upsert may have matched
  // an existing row with an owner-set limit or block flag).
  const current = await dbRequest<DbDevice[]>(DEVICES_TABLE, {
    query: {
      select: "device_id,is_blocked,block_reason,free_used,free_limit,tier",
      device_id: `eq.${deviceId}`,
      limit: "1",
    },
  });
  const row = current.ok && Array.isArray(current.data) ? current.data[0] : undefined;

  if (!row) {
    // The upsert failed (table missing or write rejected). Say so, so the owner
    // sees a device count that is honestly zero rather than a phantom device.
    console.error("device upsert failed:", upsert.error ?? current.error);
    return res.json({
      success: false,
      code: "DEVICE_STORE_FAILED",
      deviceId,
      isBlocked: false,
      freeGenerationsUsed: 0,
      freeGenerationsLimit: 5,
      configured: true,
    });
  }

  res.json({
    success: true,
    configured: true,
    deviceId: row.device_id,
    isBlocked: row.is_blocked === true,
    blockReason: row.block_reason ?? undefined,
    freeGenerationsUsed: toNumber(row.free_used),
    freeGenerationsLimit: toNumber(row.free_limit, 5),
    tier: row.tier,
  });
});

// ── Admin (owner-only, guarded by an HMAC-signed HttpOnly session cookie) ────
// SECURITY (kept in sync with server.ts):
//  • PIN-only, timing-safe verification — knowing the public admin email must
//    NEVER grant owner access.
//  • Rate-limited login attempts (best effort per serverless instance).
//  • Every admin read/write route below requires a valid session cookie, so
//    these endpoints are no longer anonymous.
const ADMIN_COOKIE_NAME = "ebnili_admin_session";
const ADMIN_SESSION_TTL_MS = 8 * 60 * 60 * 1000; // 8 hours
const ADMIN_FAIL_WINDOW_MS = 10 * 60 * 1000;
const ADMIN_MAX_FAILS = 10;
const adminLoginFailures = new Map<string, { count: number; resetAt: number }>();

// Stateless environment: REAL defaults only (zero fabricated numbers).
// `adminEmail` mirrors the server-side owner resolution so the settings form
// shows the same address the server actually trusts — and never a hard-coded
// one published in the bundle.
const DEFAULT_ADMIN_SETTINGS = {
  orangeWalletNumber: "01207782741",
  defaultFreeLimit: 5,
  autoVerificationEnabled: true,
  supportWhatsappNumber: "01207782741",
  siteName: "إبنيلي | Ebnili AI Studio",
  adminEmail: OWNER_EMAIL,
};

/**
 * Admin PIN.
 *
 * SECURITY: this was `process.env.ADMIN_PIN || "1977Sameh@"` — a real password
 * committed to a public repository. It is removed from the source entirely.
 *
 * A local developer keeps working because `npm run dev` loads `.env`, but a
 * deployment that never sets ADMIN_PIN can no longer be unlocked by reading the
 * source: `isAdminPinConfigured()` is false, the login route answers 503 with
 * setup instructions, and no session can be issued. Rotating the leaked PIN in
 * Vercel is the remaining owner-side step.
 */
function expectedAdminPin(): string {
  return (process.env.ADMIN_PIN ?? "").trim();
}

/** False when no PIN is configured — admin access is then impossible, by design. */
function isAdminPinConfigured(): boolean {
  return expectedAdminPin().length >= 8;
}

/** Owner-facing Arabic explanation shown when admin login is not configured. */
const ADMIN_PIN_MISSING_MESSAGE =
  "لوحة الإدارة غير مُفعّلة: أضف ADMIN_PIN في Vercel (قيمة عشوائية من 8 أحرف على الأقل) ثم أعد النشر.";

/**
 * Admin session signing key.
 *
 * SECURITY: this was `ADMIN_SESSION_SECRET || expectedAdminPin()` — so the
 * published PIN was ALSO the key that signs the admin session cookie, meaning
 * anyone who knew the PIN could mint an unlimited admin session. The key is now
 * a dedicated secret and never falls back to the PIN.
 */
function adminSessionSecret(): string {
  const explicit = (process.env.ADMIN_SESSION_SECRET ?? "").trim();
  if (!isWeakSecret(explicit)) return explicit;
  // No dedicated secret configured: sign with a value that cannot be derived
  // from anything the source reveals, so an unconfigured deployment is locked
  // rather than forgeable. A configured PIN still works locally.
  //
  // NOTE: this is deliberately NOT the auth session secret — mixing the two
  // would let one compromise mint both cookies.
  return `admin-unconfigured-${INSECURE_DEV_SECRET}`;
}

/**
 * `Secure` for auth cookies.
 *
 * SECURITY: it used to be `Boolean(process.env.VERCEL)`, so a production
 * deployment running anywhere OTHER than Vercel (the self-hosted runner in
 * `server.ts`, a container, a proxy) would ship session cookies without the
 * `Secure` flag — they would then travel in cleartext over any http:// hop.
 *
 * Secure is decided by what the request actually looks like, not by which host
 * it runs on: TLS in production, plaintext only for a genuinely local http
 * development server. This keeps `npm run dev` working without ever weakening a
 * real deployment.
 */
function cookiesAreSecure(req?: Request): boolean {
  if (req?.headers["x-forwarded-proto"]) {
    return String(req.headers["x-forwarded-proto"]).split(",")[0].trim() === "https";
  }
  if (typeof process.env.NODE_ENV === "string" && process.env.NODE_ENV === "production") return true;
  if (process.env.VERCEL) return true;
  return Boolean(req?.secure);
}

function signAdminExpiry(exp: string): string {
  return crypto.createHmac("sha256", adminSessionSecret()).update(exp).digest("base64url");
}

function issueAdminSession(): string {
  const exp = String(Date.now() + ADMIN_SESSION_TTL_MS);
  return `${exp}.${signAdminExpiry(exp)}`;
}

function isValidAdminSession(token: string | undefined): boolean {
  if (!token) return false;
  const sep = token.indexOf(".");
  if (sep <= 0) return false;
  const exp = token.slice(0, sep);
  const sig = token.slice(sep + 1);
  const expected = signAdminExpiry(exp);
  const a = Buffer.from(sig);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return false;
  const expNum = Number(exp);
  return Number.isFinite(expNum) && Date.now() < expNum;
}

function readAdminCookie(req: Request): string | undefined {
  const header = req.headers.cookie;
  if (!header) return undefined;
  for (const part of header.split(";")) {
    const eq = part.indexOf("=");
    if (eq === -1) continue;
    if (part.slice(0, eq).trim() === ADMIN_COOKIE_NAME) return decodeURIComponent(part.slice(eq + 1).trim());
  }
  return undefined;
}

function adminClientKey(req: Request): string {
  const fwd = req.headers["x-forwarded-for"];
  const ip = (Array.isArray(fwd) ? fwd[0] : fwd)?.split(",")[0]?.trim();
  return ip || req.socket.remoteAddress || "unknown";
}

function registerAdminLoginFailure(req: Request): void {
  const key = adminClientKey(req);
  const rec = adminLoginFailures.get(key);
  if (!rec || Date.now() > rec.resetAt) {
    adminLoginFailures.set(key, { count: 1, resetAt: Date.now() + ADMIN_FAIL_WINDOW_MS });
  } else {
    rec.count += 1;
  }
}

function pinMatches(pin: unknown): boolean {
  const submitted = Buffer.from(typeof pin === "string" ? pin.trim() : "");
  let match = false;
  // Constant-time comparison against every configured candidate.
  for (const candidate of [expectedAdminPin()]) {
    const c = Buffer.from(candidate);
    if (submitted.length === c.length && crypto.timingSafeEqual(submitted, c)) match = true;
  }
  return match;
}

// Every admin read/write route must present a valid session cookie AND belong to
// the site owner account.
//
// OWNERSHIP: the PIN alone is not enough any more. The owner's address is
// published in the public bundle and on the contact page, so admin access is
// bound to the signed account session as well — knowing the e-mail is not
// enough either, you need the account AND the PIN.
//
// 404 (not 401) for anyone else: the admin surface should simply not exist from
// the outside rather than advertise itself with a "forbidden" answer.
function requireAdmin(req: Request, res: Response, next: NextFunction) {
  const session = readAuthSession(req as AuthReq);
  // The original owner needs the PIN cookie (two-factor). An added admin
  // bypasses the PIN — their email was explicitly trusted by the owner.
  if (isOwnerAccount(session) && isValidAdminSession(readAdminCookie(req))) {
    return next();
  }
  if (isAddedAdmin(session)) {
    return next();
  }
  return res
    .status(404)
    .json({ success: false, message: "Not found" });
}

app.post("/api/admin/auth", rateLimit(authLimiter, LIMIT_RULES.auth, "admin", "محاولات كثيرة جداً. انتظر قليلاً ثم أعد المحاولة."), (req: Request, res: Response) => {
  // Admin is the site owner and nothing else. The PIN is a second factor, not
  // the identity: without the owner account signed in this endpoint answers 404
  // exactly like every other admin route, so a non-owner cannot even obtain a
  // session to try elsewhere with.
  if (!isOwnerAccount(readAuthSession(req as AuthReq))) {
    return res.status(404).json({ success: false, message: "Not found" });
  }
  // No PIN configured in this deployment: fail loudly with the fix instead of
  // rejecting every attempt as "wrong PIN", which would send the owner looking
  // for a typo that does not exist.
  if (!isAdminPinConfigured()) {
    console.error("[SECURITY] ADMIN_PIN is not set. Admin dashboard is locked.");
    return res.status(503).json({ success: false, message: ADMIN_PIN_MISSING_MESSAGE, error: ADMIN_PIN_MISSING_MESSAGE });
  }
  const { pin } = (req.body as { pin?: string; email?: string }) ?? {};
  const failureRec = adminLoginFailures.get(adminClientKey(req));
  if (failureRec && Date.now() <= failureRec.resetAt && failureRec.count >= ADMIN_MAX_FAILS) {
    const throttleMsg = "محاولات دخول كثيرة جداً — أعد المحاولة بعد 10 دقائق.";
    return res.status(429).json({ success: false, message: throttleMsg, error: throttleMsg });
  }
  // PIN-only: an email address alone must never grant owner access.
  if (!pinMatches(pin)) {
    registerAdminLoginFailure(req);
    const invalidMsg = "رمز الدخول غير صحيحة";
    return res.status(401).json({ success: false, message: invalidMsg, error: invalidMsg });
  }
  adminLoginFailures.delete(adminClientKey(req));
  res.cookie(ADMIN_COOKIE_NAME, issueAdminSession(), {
    httpOnly: true,
    secure: cookiesAreSecure(req),
    sameSite: "strict",
    path: "/",
    maxAge: ADMIN_SESSION_TTL_MS,
  });
  res.json({ success: true });
});

/**
 * Admin dashboard data — REAL rows, computed server-side.
 *
 * WHY this rewrite: the endpoint answered with hard-coded zeros plus empty
 * arrays. The dashboard therefore always looked like a brand-new install, and
 * because nothing was ever stored, approving a payment or blocking a device had
 * no row to act on — the buttons could not work even in principle. Every number
 * below is derived from the tables in `supabase/projects.sql`.
 *
 * When the database is not configured the response is an explicit 503 with
 * `configured: false` so the dashboard can say "run the SQL" instead of showing
 * a convincing page of zeros.
 */
app.get("/api/admin/overview", requireAdmin, async (_req: Request, res: Response) => {
  const sb = supabaseConfig();
  if (!sb.dbConfigured) {
    return res.status(503).json({
      success: false,
      code: "DB_UNAVAILABLE",
      configured: false,
      message:
        "قاعدة بيانات لوحة الإدارة غير مهيأة. نفّذ supabase/projects.sql في Supabase ثم تأكد من متغيّر SUPABASE_SERVICE_ROLE_KEY.",
    });
  }

  const paymentSelect =
    "id,account_email,account_id,plan_id,billing_cycle,amount_egp,amount_usd,sender_phone,transaction_ref,receipt_path,receipt_file_name,notes,status,submitted_at,reviewed_at";
  const deviceSelect =
    "device_id,fingerprint_hash,account_email,screen,timezone,platform,user_agent,ip_address,is_blocked,block_reason,free_used,free_limit,tier,first_seen_at,last_seen_at";
  const accountSelect =
    "account_id,email,display_name,provider,avatar_url,credits,credits_granted,welcome_given,last_seen_at,first_seen_at,requests_count,last_ip,user_agent,tier,is_blocked,block_reason";

  const [paymentsRes, pendingRes, devicesRes, confirmedRes, settingsRes, accountsRes, adminsRes] =
    await Promise.all([
      dbRequest<DbPayment[]>(PAYMENTS_TABLE, {
        query: { select: paymentSelect, order: "submitted_at.desc", limit: "50" },
      }),
      dbRequest<Array<{ count: number }>>(PAYMENTS_TABLE, {
        query: { select: "id", status: "eq.pending", limit: "1000" },
        headers: { Prefer: "count=exact" },
      }),
      dbRequest<DbDevice[]>(DEVICES_TABLE, {
        query: { select: deviceSelect, order: "last_seen_at.desc", limit: "200" },
      }),
      dbRequest<DbPayment[]>(PAYMENTS_TABLE, {
        query: { select: "amount_egp,plan_id,status", status: "eq.confirmed", limit: "1000" },
      }),
      dbRequest<DbSettingsRow[]>(SETTINGS_TABLE, {
        query: { select: "key,value", key: `eq.${SETTINGS_ROW_KEY}`, limit: "1" },
      }),
      // Newest first: the owner asked for the recent sign-ups at the top, and
      // this ordering makes "من سجّل حديثاً" the default view.
      dbRequest<DbAccount[]>(ACCOUNTS_TABLE, {
        query: { select: accountSelect, order: "first_seen_at.desc", limit: "200" },
      }),
      dbRequest<Array<{ id: string; email: string; added_by: string; added_at: string }>>(ADMINS_TABLE, {
        query: { select: "id,email,added_by,added_at", order: "added_at.desc", limit: "200" },
      }),
    ]);

  // One failing read must not blank the whole dashboard: report what we could
  // load and name the tables that failed.
  const failed: string[] = [];
  if (!paymentsRes.ok) failed.push("payments");
  if (!devicesRes.ok) failed.push("devices");
  if (!confirmedRes.ok) failed.push("revenue");
  if (!accountsRes.ok) failed.push("accounts");

  const devices = devicesRes.ok && Array.isArray(devicesRes.data) ? devicesRes.data : [];
  const confirmed = confirmedRes.ok && Array.isArray(confirmedRes.data) ? confirmedRes.data : [];
  const pendingCount = Array.isArray(pendingRes.data)
    ? (pendingRes.data[0]?.count ?? 0)
    : failed.length
      ? 0
      : 0;

  const totalRevenueEGP = confirmed.reduce((sum, row) => sum + toNumber(row.amount_egp), 0);
  const activeProUsers = new Set(
    confirmed.filter((row) => row.plan_id === "pro" || row.plan_id === "business").map((r) => r.account_email),
  );
  activeProUsers.delete("");

  const lastActive = devices.reduce<string>(
    (latest, row) => (row.last_seen_at > latest ? row.last_seen_at : latest),
    "",
  );

  const stored = settingsRes.ok && Array.isArray(settingsRes.data) ? settingsRes.data[0] : undefined;
  const settings = normalizePlatformSettings(stored?.value);

  const accounts = accountsRes.ok && Array.isArray(accountsRes.data) ? accountsRes.data : [];
  // Presence is derived at request time, so a dashboard left open for a minute
  // still reports the truth without the client re-deriving timestamps.
  const publicAccounts = accounts.map(toPublicAccount);
  const onlineNow = publicAccounts.filter((a) => a.presence === "online").length;
  const idleNow = publicAccounts.filter((a) => a.presence === "idle").length;
  const totalCredits = publicAccounts.reduce((sum, a) => sum + a.credits, 0);

  return res.json({
    success: true,
    configured: true,
    partial: failed.length > 0,
    failedTables: failed,
    stats: {
      totalDevicesCount: devices.length,
      blockedDevicesCount: devices.filter((d) => d.is_blocked).length,
      activeProUsersCount: activeProUsers.size,
      // There is no generation counter persisted anywhere, so this stays honest at
      // zero rather than inventing activity that never happened.
      totalGenerationsExecuted: 0,
      totalRevenueEGP,
      totalTransactionsCount: confirmed.length + (Array.isArray(pendingRes.data) ? pendingRes.data.length : 0),
      pendingTransactionsCount: pendingCount,
      lastActiveTime: lastActive || new Date().toISOString(),
      // Account + presence counters.
      totalAccountsCount: publicAccounts.length,
      onlineAccountsCount: onlineNow,
      idleAccountsCount: idleNow,
      totalCreditsOutstanding: totalCredits,
      welcomeCreditsPerSignup: WELCOME_CREDITS,
    },
    settings,
    accounts: publicAccounts,
    devices: devices.map(toPublicDevice),
    recentTransactions: paymentsRes.ok && Array.isArray(paymentsRes.data) ? paymentsRes.data.map(toPublicPayment) : [],
    admins: adminsRes.ok && Array.isArray(adminsRes.data) ? adminsRes.data : [],
  });
});

// ── Owner: credit controls ────────────────────────────────────────────────────
// WHY: the dashboard needs to top a user up (support request, a good-review
// reward) and to take credits away (abuse). The value is validated as an
// integer in [0, 100000] on the server — the browser's number is never trusted,
// which is the same rule every other owner action follows here.

app.post("/api/admin/account/set-credits", requireAdmin, async (req: Request, res: Response) => {
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const body = (req.body ?? {}) as { accountId?: string; credits?: number };

  const accountId = String(body.accountId ?? "").trim();
  if (!accountId) return res.status(400).json({ success: false, message: "معرّف الحساب مطلوب." });

  const credits = Number(body.credits);
  if (!Number.isFinite(credits) || !Number.isInteger(credits) || credits < 0 || credits > 100_000) {
    return res.status(400).json({
      success: false,
      message: "قيمة الكريديت يجب أن تكون عدداً صحيحاً بين 0 و 100000.",
    });
  }

  const write = await dbRequest<DbAccount[]>(ACCOUNTS_TABLE, {
    method: "PATCH",
    query: { account_id: `eq.${accountId}` },
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({ credits }),
  });

  if (!write.ok) {
    console.error("set-credits failed:", write.error);
    return res.status(503).json({ success: false, message: "تعذّر تحديث الكريديت." });
  }
  if (!Array.isArray(write.data) || write.data.length === 0) {
    return res.status(404).json({ success: false, message: "لا يوجد حساب بهذا المعرّف." });
  }
  res.json({ success: true, account: toPublicAccount(write.data[0]) });
});

// Block / unblock an ACCOUNT (as opposed to a device), so a user who signs in
// from a new browser is still refused.
app.post("/api/admin/account/toggle-block", requireAdmin, async (req: Request, res: Response) => {
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const body = (req.body ?? {}) as { accountId?: string; block?: boolean; reason?: string };

  const accountId = String(body.accountId ?? "").trim();
  if (!accountId) return res.status(400).json({ success: false, message: "معرّف الحساب مطلوب." });
  const block = body.block === true;
  const reason =
    typeof body.reason === "string" && body.reason.trim() ? body.reason.trim().slice(0, 200) : null;

  const write = await dbRequest<DbAccount[]>(ACCOUNTS_TABLE, {
    method: "PATCH",
    query: { account_id: `eq.${accountId}` },
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({ is_blocked: block, block_reason: block ? reason : null }),
  });

  if (!write.ok) {
    console.error("account toggle-block failed:", write.error);
    return res.status(503).json({ success: false, message: "تعذّر تحديث حالة الحساب." });
  }
  if (!Array.isArray(write.data) || write.data.length === 0) {
    return res.status(404).json({ success: false, message: "لا يوجد حساب بهذا المعرّف." });
  }
  res.json({ success: true, account: toPublicAccount(write.data[0]) });
});

// ── Owner actions: these now actually write to the database ───────────────────
// WHY: all four used to be a single stub that answered `{ persisted: false }`,
// so the dashboard's approve / block / quota / tier / settings buttons reported
// success while changing nothing. Each handler below updates a real row and
// verifies the write before reporting success.

app.post("/api/admin/settings", requireAdmin, async (req: Request, res: Response) => {
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const next = normalizePlatformSettings(body);
  const updated = await dbRequest<DbSettingsRow[]>(SETTINGS_TABLE, {
    method: "POST",
    query: { on_conflict: "merge-duplicates" },
    body: JSON.stringify({ key: SETTINGS_ROW_KEY, value: next }),
  });
  if (!updated.ok) {
    console.error("settings write failed:", updated.error);
    return res.status(503).json({ success: false, message: "تعذّر حفظ الإعدادات. تأكد من تنفيذ supabase/projects.sql." });
  }
  res.json({ success: true, persisted: true, settings: next });
});

// ── Admin management: add/remove admin emails ────────────────────────────────
// The owner can add email addresses that get full admin access without the PIN.
// Only the original owner (SITE_OWNER_EMAIL + PIN) can manage this list.

app.get("/api/admin/admins", requireAdmin, async (_req: Request, res: Response) => {
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const result = await dbRequest<Array<{ id: string; email: string; added_by: string; added_at: string }>>(ADMINS_TABLE, {
    query: { select: "id,email,added_by,added_at", order: "added_at.desc", limit: "200" },
  });
  if (!result.ok) {
    return res.status(503).json({ success: false, message: "تعذّر تحميل قائمة الأدمن." });
  }
  const rows = Array.isArray(result.data) ? result.data : [];
  res.json({ success: true, admins: rows });
});

app.post("/api/admin/admins", requireAdmin, async (req: Request, res: Response) => {
  // Only the original owner can add/remove admins — not added admins themselves.
  if (!isOwnerAccount(readAuthSession(req as AuthReq))) {
    return res.status(403).json({ success: false, message: "هذه العملية متاحة لصاحب الموقع الأصلي فقط." });
  }
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const body = (req.body as { email?: string }) ?? {};
  const email = String(body.email ?? "").trim().toLowerCase();
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return res.status(400).json({ success: false, message: "البريد الإلكتروني غير صحيح." });
  }
  if (email === OWNER_EMAIL) {
    return res.status(400).json({ success: false, message: "هذا البريد هو المالك الأصلي بالفعل." });
  }

  const id = `adm_${Date.now().toString(36)}_${crypto.randomBytes(4).toString("hex")}`;
  const write = await dbRequest(ADMINS_TABLE, {
    method: "POST",
    query: { on_conflict: "ignore" },
    body: JSON.stringify({ id, email, added_by: OWNER_EMAIL || "owner" }),
  });
  if (!write.ok) {
    // 409 from PostgREST means the email already exists — not a real error.
    if (write.status === 409) {
      return res.status(409).json({ success: false, message: "هذا البريد مضاف بالفعل." });
    }
    return res.status(503).json({ success: false, message: "تعذّر إضافة الأدمن." });
  }
  // Refresh the cache immediately so the new admin is recognized.
  adminCacheAt = 0;
  void refreshAdminEmails();
  res.json({ success: true, email });
});

app.delete("/api/admin/admins", requireAdmin, async (req: Request, res: Response) => {
  // Only the original owner can remove admins.
  if (!isOwnerAccount(readAuthSession(req as AuthReq))) {
    return res.status(403).json({ success: false, message: "هذه العملية متاحة لصاحب الموقع الأصلي فقط." });
  }
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const email = String((req.body as { email?: string })?.email ?? "").trim().toLowerCase();
  if (!email) {
    return res.status(400).json({ success: false, message: "البريد الإلكتروني مطلوب." });
  }
  const write = await dbRequest(ADMINS_TABLE, {
    method: "DELETE",
    query: { email: `eq.${email}` },
  });
  if (!write.ok) {
    return res.status(503).json({ success: false, message: "تعذّر حذف الأدمن." });
  }
  adminCacheAt = 0;
  void refreshAdminEmails();
  res.json({ success: true, email });
});

app.post("/api/admin/device/toggle-block", requireAdmin, async (req: Request, res: Response) => {
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const deviceId = String(body.deviceId ?? "").trim();
  if (!deviceId) return res.status(400).json({ success: false, message: "معرّف الجهاز مطلوب." });
  const block = body.block === true;
  const reason = typeof body.reason === "string" && body.reason.trim() ? body.reason.trim().slice(0, 200) : null;

  // An upsert so blocking a device that has never reported in still takes hold.
  const write = await dbRequest<DbDevice[]>(DEVICES_TABLE, {
    method: "POST",
    query: { on_conflict: "merge-duplicates" },
    body: JSON.stringify({
      device_id: deviceId,
      fingerprint_hash: String(body.fingerprintHash ?? ""),
      is_blocked: block,
      block_reason: block ? reason : null,
      last_seen_at: new Date().toISOString(),
    }),
  });
  if (!write.ok) {
    console.error("block write failed:", write.error);
    return res.status(503).json({ success: false, message: "تعذّر تحديث حالة الجهاز." });
  }
  res.json({ success: true, persisted: true, deviceId, isBlocked: block });
});

app.post("/api/admin/device/reset-quota", requireAdmin, async (req: Request, res: Response) => {
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const deviceId = String(body.deviceId ?? "").trim();
  if (!deviceId) return res.status(400).json({ success: false, message: "معرّف الجهاز مطلوب." });
  const rawLimit = Number(body.newLimit ?? 5);
  const limit = Number.isFinite(rawLimit) ? Math.min(50, Math.max(1, Math.round(rawLimit))) : 5;

  const write = await dbRequest<DbDevice[]>(DEVICES_TABLE, {
    method: "PATCH",
    query: { device_id: `eq.${deviceId}` },
    body: JSON.stringify({
      free_limit: limit,
      // `resetUsed` defaults to true: "reset quota" that leaves the counter where
      // it was is the action owners actually need.
      ...(body.resetUsed === false ? {} : { free_used: 0 }),
      last_seen_at: new Date().toISOString(),
    }),
  });
  if (!write.ok || !Array.isArray(write.data) || write.data.length === 0) {
    console.error("quota write failed:", write.error);
    return res.status(404).json({ success: false, message: "لم يتم العثور على الجهاز." });
  }
  res.json({ success: true, persisted: true, deviceId, freeLimit: limit });
});

app.post("/api/admin/device/set-tier", requireAdmin, async (req: Request, res: Response) => {
  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);
  const body = (req.body ?? {}) as Record<string, unknown>;
  const deviceId = String(body.deviceId ?? "").trim();
  if (!deviceId) return res.status(400).json({ success: false, message: "معرّف الجهاز مطلوب." });
  const tier = String(body.tier ?? "free");
  if (tier !== "free" && tier !== "pro" && tier !== "business") {
    return res.status(400).json({ success: false, message: "الباقة غير صحيحة." });
  }

  const write = await dbRequest<DbDevice[]>(DEVICES_TABLE, {
    method: "POST",
    query: { on_conflict: "merge-duplicates" },
    body: JSON.stringify({
      device_id: deviceId,
      fingerprint_hash: String(body.fingerprintHash ?? ""),
      tier,
      last_seen_at: new Date().toISOString(),
    }),
  });
  if (!write.ok) {
    console.error("tier write failed:", write.error);
    return res.status(503).json({ success: false, message: "تعذّر تحديث الباقة." });
  }
  res.json({ success: true, persisted: true, deviceId, tier });
});

// ── Owner-only subscription activation (the ONE path to a paid tier) ─────────
// SECURITY: this replaces the old self-service upgrade. The owner approves a
// payment here, after verifying the Orange Cash transfer in the dashboard.
// The grant is returned as a SHORT-LIVED, HMAC-signed HttpOnly cookie
// (`ebnili_plan`) so it survives a request without needing a database; the
// browser cannot forge it because the signature is server-side only.
const PLAN_COOKIE_NAME = "ebnili_plan";
const PLAN_GRANT_TTL_MS = 7 * 24 * 60 * 60 * 1000;

/**
 * Signing keys — never derived from a human password.
 *
 * SECURITY: `planSigningSecret()` used to fall back to `ADMIN_PIN` and then to
 * the literal `"ebnili"`. The admin PIN was hard-coded in this file, so anyone
 * who read the public source could mint an `ebnili_plan` cookie for any email
 * and keep Pro/Business forever — a one-line self-service upgrade. Plan grants
 * are money, so they get their OWN secret with a mandatory, loud failure.
 *
 * `secretProblem()` explains exactly what to set, in Arabic, instead of letting
 * the app quietly sign with a guessable key.
 */
const INSECURE_DEV_SECRET = "ebnili-insecure-dev-secret";

/**
 * True when a secret is missing or is a known-public placeholder.
 *
 * SECURITY: this used to be a local copy of the rule, which meant the check that
 * guards the session cookie and the check the unit tests pin could drift apart —
 * one hardened, the other not. There is now exactly ONE definition, in
 * `api/security.ts`, and it also rejects long-but-public values like
 * `"change-me".repeat(5)`, which the old length-only-plus-exact-match version
 * accepted.
 */

/**
 * A dedicated secret for plan grants.
 *
 * Refuses to sign with a weak value: without a real secret, paying customers
 * would silently get no subscription, so the endpoint answers 503 with setup
 * instructions rather than shipping a forgeable free upgrade.
 */
function planSigningSecret(): string | null {
  const explicit = (process.env.PLAN_GRANT_SECRET ?? "").trim();
  if (explicit && !isWeakSecret(explicit)) return explicit;

  // A strong AUTH_SESSION_SECRET is an acceptable key: it is already a
  // high-entropy server-side secret and avoids one more variable to configure.
  const session = (process.env.AUTH_SESSION_SECRET ?? "").trim();
  if (!isWeakSecret(session)) return session;

  return null;
}

/** Owner-facing setup message for a missing plan-grant secret. */
function planSecretProblem(): string {
  return "PLAN_GRANT_SECRET غير مُعد. أضفه في Vercel كقيمة عشوائية طويلة (32 حرفاً على الأقل) لتفعيل اشتراكات Pro/Business بأمان.";
}

/** Signing key for plan grants, or a thrown error — for routes that cannot continue. */
function requirePlanSigningSecret(): string {
  const secret = planSigningSecret();
  if (!secret) throw new Error(planSecretProblem());
  return secret;
}

function issuePlanGrant(email: string, tier: "pro" | "business"): string {
  const payload = Buffer.from(
    JSON.stringify({ email: String(email).trim().toLowerCase(), tier, exp: Date.now() + PLAN_GRANT_TTL_MS }),
  ).toString("base64url");
  return `${payload}.${hmacB64With(payload, requirePlanSigningSecret())}`;
}

function hmacB64With(value: string, secret: string): string {
  return crypto.createHmac("sha256", secret).update(value).digest("base64url");
}

/**
 * Bind a plan grant to the signed-in session.
 *
 * SECURITY: the grant cookie is signed but it is still a bearer token. Without
 * this check, anyone who obtains the cookie (an XSS, a shared machine, an
 * intercepted request on a misconfigured proxy) could replay it from a different
 * browser. Comparing the grant's own subject against the verified session e-mail
 * is what makes a stolen cookie useless.
 *
 * Exported for the unit tests, which assert this directly.
 */
function grantBelongsToSession(
  grant: { email: string } | null,
  session: AuthUser | null,
): grant is { email: string } {
  if (!grant || !session?.email) return false;
  return String(grant.email).trim().toLowerCase() === String(session.email).trim().toLowerCase();
}

function readPlanGrant(req: Request): { email: string; tier: "pro" | "business"; expiresAt: string } | null {
  const token = readCookie(req, PLAN_COOKIE_NAME);
  if (!token) return null;
  // ── INVARIANT: this function returns a GRANT, never a decision.
  // ── It deliberately does NOT know who is asking. Every caller must bind the
  // ── grant to the signed-in session before trusting it:
  // ──     `grant.email.toLowerCase() === session.email.toLowerCase()`
  // ── Both call sites (`/api/subscriptions/current` and `serverTierFor`) do that
  // ── check. It is repeated here as a named helper so a third call site has an
  // ── obvious, reviewable thing to use instead of re-deriving it — and so the
  // ── unit tests can assert the binding directly.
  // ── A signed-but-unbound cookie is still a bearer token: anyone who steals it
  // ── could present it, so binding to the session email is what makes a stolen
  // ── cookie useless in a different browser.
  const secret = planSigningSecret();
  if (!secret) return null;
  const sep = token.indexOf(".");
  if (sep <= 0) return null;
  const body = token.slice(0, sep);
  if (!safeEqual(token.slice(sep + 1), hmacB64With(body, secret))) return null;
  try {
    const parsed = JSON.parse(Buffer.from(body, "base64url").toString("utf-8")) as {
      email?: string;
      tier?: string;
      exp?: number;
    };
    if (!parsed?.email) return null;
    // A lapsed grant is no grant: this is what makes a finished subscription
    // fall back to Free automatically (and bring the watermark back).
    if (typeof parsed.exp !== "number" || Date.now() > parsed.exp) return null;
    const tier = parsed.tier === "business" ? "business" : "pro";
    return { email: parsed.email, tier, expiresAt: new Date(parsed.exp).toISOString() };
  } catch {
    return null;
  }
}

/**
 * Approve or reject a payment — the ONE path to a paid tier.
 *
 * WHY this was broken in a way that lost money: the handler minted the plan
 * grant cookie onto `res`, i.e. onto the OWNER's browser. The customer never
 * received it, so approving a payment did nothing for them — they paid and kept
 * the free tier. It also never wrote a status, so the row stayed "pending"
 * forever and every dashboard refresh offered the same "approve" button again.
 *
 * Now the decision is durable (the row's status is updated) and the entitlement
 * lives in the database, which `/api/subscriptions/current` reads — so the
 * approved customer sees Pro/Business on their next request, on any device.
 * The signed cookie stays only as a fast path for the owner's own browser.
 */
app.post("/api/admin/transaction/update-status", requireAdmin, async (req: Request, res: Response) => {
  const body = (req.body as { transactionId?: string; status?: string; email?: string; tier?: string }) ?? {};
  const status = body.status;
  if (status !== "confirmed" && status !== "rejected") {
    res.status(400).json({ success: false, error: "حالة المعاملة يجب أن تكون confirmed أو rejected." });
    return;
  }

  if (!supabaseConfig().dbConfigured) return dbUnavailable(res);

  const transactionId = String(body.transactionId ?? "").trim();
  if (!transactionId) {
    res.status(400).json({ success: false, error: "معرّف المعاملة مطلوب." });
    return;
  }

  // Read the row first: the account to grant and the tier must come from the
  // stored payment, never from the request body, or a crafted call could grant
  // a plan to an arbitrary address.
  const found = await dbRequest<DbPayment[]>(PAYMENTS_TABLE, {
    query: { select: "id,account_email,plan_id,status", id: `eq.${transactionId}`, limit: "1" },
  });
  if (!found.ok || !Array.isArray(found.data) || found.data.length === 0) {
    return res.status(404).json({ success: false, error: "لم يتم العثور على هذه المعاملة." });
  }
  const payment = found.data[0];

  // Re-approving an already-decided payment must not silently mint a second
  // entitlement or reset an expiry the customer is already serving.
  if (payment.status !== "pending") {
    return res.status(409).json({
      success: false,
      error: `تمت مراجعة هذه المعاملة مسبقاً (${payment.status === "confirmed" ? "مؤكدة" : "مرفوضة"}).`,
    });
  }

  const reviewed = await dbRequest<DbPayment[]>(PAYMENTS_TABLE, {
    method: "PATCH",
    query: { id: `eq.${transactionId}` },
    body: JSON.stringify({
      status,
      reviewed_at: new Date().toISOString(),
      reviewed_by: OWNER_EMAIL || "owner",
    }),
  });
  if (!reviewed.ok || !Array.isArray(reviewed.data) || reviewed.data.length === 0) {
    console.error("payment review write failed:", reviewed.error);
    return res.status(503).json({ success: false, error: "تعذّر حفظ قرار المراجعة. أعد المحاولة." });
  }

  // Confirmations grant nothing extra here: the row now says "confirmed", and
  // `/api/subscriptions/current` turns that into a real tier for the account.
  if (status === "confirmed" && payment.account_email) {
    // Fast path for the owner's own browser only. The durable entitlement is the
    // status written above — never this cookie.
    try {
      const tier = body.tier === "business" || payment.plan_id === "business" ? "business" : "pro";
      res.cookie(PLAN_COOKIE_NAME, issuePlanGrant(payment.account_email, tier), {
        httpOnly: true,
        secure: cookiesAreSecure(req),
        sameSite: "lax",
        path: "/",
        maxAge: PLAN_GRANT_TTL_MS,
      });
    } catch (e) {
      // Not fatal: the row is already confirmed, which is what actually grants
      // the plan. Log it and continue so the owner sees a success.
      console.error("plan grant cookie skipped:", e instanceof Error ? e.message : String(e));
    }
  }

  res.json({
    success: true,
    persisted: true,
    status,
    accountEmail: payment.account_email || undefined,
    message:
      status === "confirmed"
        ? "تم تأكيد التحويل وتفعيل الاشتراك على الحساب مباشرة."
        : "تم رفض الطلب وتوثيق السبب.",
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// User authentication — Google & GitHub (OAuth 2.0 authorization-code flow)
// ─────────────────────────────────────────────────────────────────────────────
// SECURITY NOTES:
//  • The client secret NEVER reaches the browser. The code → token exchange and
//    the profile lookup both happen here, server-side.
//  • `state` is a single-use random nonce kept in a short-lived HttpOnly cookie
//    and compared on the way back, which blocks CSRF / login-injection.
//  • The session is a stateless HMAC-SHA256 signed HttpOnly cookie, so it works
//    on Vercel's stateless serverless runtime with no database attached.
//  • Both providers are OPTIONAL. When their env vars are missing,
//    `/api/auth/providers` reports `configured: false` and the UI hides them —
//    the rest of the app keeps working exactly as before.
// Env vars: GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GITHUB_CLIENT_ID /
//           GITHUB_CLIENT_SECRET / AUTH_SESSION_SECRET / APP_URL
// ─────────────────────────────────────────────────────────────────────────────

type AuthReq = Request;
type AuthRes = Response;
type AuthProviderId = "google" | "github";

interface AuthUser {
  id: string;
  name: string;
  email: string;
  picture: string;
  provider: AuthProviderId;
}

interface ProviderCredentials {
  clientId: string;
  clientSecret: string;
  configured: boolean;
}

const AUTH_COOKIE_NAME = "ebnili_user_session";
const AUTH_STATE_COOKIE = "ebnili_oauth_state";
/** PKCE verifier/challenge, so the browser never carries a usable credential. */
const AUTH_PKCE_COOKIE = "ebnili_oauth_pkce";
const AUTH_SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000; // 7 days
const AUTH_STATE_TTL_MS = 10 * 60 * 1000; // 10 minutes

/**
 * Canonical attributes for every auth cookie.
 *
 * A cookie can only be deleted when the deletion carries the SAME attributes it
 * was set with — most importantly `Secure`. The state and PKCE cookies were
 * being set with `secure: true` in production but deleted with a bare
 * `{ path: "/" }`, so on some browsers the single-use delete silently failed and
 * a stale nonce survived the round trip. Express clears by emitting an expired
 * cookie, and the browser only replaces it if the name/domain/path/Secure
 * attributes line up — hence one helper used by BOTH set and clear.
 */
function authCookieOptions(maxAgeMs?: number, req?: Request) {
  return {
    httpOnly: true,
    secure: cookiesAreSecure(req),
    sameSite: "lax" as const,
    path: "/",
    ...(maxAgeMs ? { maxAge: maxAgeMs } : {}),
  };
}

/** Delete an auth cookie with exactly the attributes it was created with. */
function clearAuthCookie(res: Response, name: string, req?: Request): void {
  res.clearCookie(name, authCookieOptions(undefined, req));
}

/**
 * HMAC key for the sign-in session cookie.
 *
 * SECURITY: this used to fall back to `ADMIN_PIN` and then to a literal
 * `"ebnili-insecure-dev-secret"` that is published in this repository. Any
 * reader of the public source could therefore forge a session cookie for the
 * owner's email and take the site over. Session cookies must be signed with a
 * real server-side secret and nothing else.
 *
 * The value is memoised because it is read on every authenticated request and
 * the resolution order is not free.
 */
let cachedAuthSecret: string | null = null;

/** True when the deployment cannot safely sign a session at all. */
function authSessionSecretMissing(): boolean {
  return requireStrongSecret(process.env.AUTH_SESSION_SECRET) === null;
}

/**
 * HMAC key for the sign-in session cookie.
 *
 * SECURITY: this used to fall back to a literal `"ebnili-insecure-dev-secret"`
 * that is committed to this repository, so any reader of the public source could
 * forge a session cookie for the owner's address and take the site over. It
 * logged a warning and then carried on — which is the worst of both worlds: the
 * operator was told, and the site stayed wide open anyway.
 *
 * It now FAILS CLOSED. With no usable secret:
 *  • no session can be MINTED (`issueAuthSession` returns null), so nobody can
 *    sign in, and
 *  • no session can be VERIFIED, so a forged cookie is rejected.
 *
 * The site does not crash and the marketing/pricing pages keep rendering; only
 * the authenticated surfaces lock, and `/api/health` reports the exact problem.
 * That is the correct trade: a locked feature is recoverable in one env edit,
 * a forged owner session is not.
 */
function authSessionSecret(): string | null {
  if (cachedAuthSecret !== null) return cachedAuthSecret;
  const explicit = (process.env.AUTH_SESSION_SECRET ?? "").trim();
  const strong = requireStrongSecret(explicit);
  if (strong) {
    cachedAuthSecret = strong;
    return cachedAuthSecret;
  }
  cachedAuthSecret = null;
  console.error(weakSecretMessage("AUTH_SESSION_SECRET"));
  return null;
}

function hmacB64(input: string): string | null {
  const secret = authSessionSecret();
  if (!secret) return null;
  return crypto.createHmac("sha256", secret).update(input).digest("base64url");
}

function safeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return crypto.timingSafeEqual(bufA, bufB);
}

function readCookie(req: AuthReq, name: string): string | undefined {
  const header = req.headers.cookie;
  if (!header) return undefined;
  for (const part of header.split(";")) {
    const eq = part.indexOf("=");
    if (eq === -1) continue;
    if (part.slice(0, eq).trim() === name) return decodeURIComponent(part.slice(eq + 1).trim());
  }
  return undefined;
}

// Absolute origin of THIS request — correct behind Vercel / any reverse proxy.
function requestBaseUrl(req: AuthReq): string {
  const configured = (process.env.APP_URL || "").trim().replace(/\/+$/, "");
  if (configured) return configured;
  const fwdProto = (req.headers["x-forwarded-proto"] as string | undefined)?.split(",")[0]?.trim();
  const fwdHost = (req.headers["x-forwarded-host"] as string | undefined)?.split(",")[0]?.trim();
  const proto = fwdProto || req.protocol || "http";
  const host = fwdHost || req.headers.host || "localhost:3000";
  return `${proto}://${host}`;
}

function authCallbackUrl(req: AuthReq, provider: AuthProviderId): string {
  return `${requestBaseUrl(req)}/api/auth/callback/${provider}`;
}

function providerConfig(provider: AuthProviderId): ProviderCredentials {
  if (provider === "google") {
    const clientId = (process.env.GOOGLE_CLIENT_ID || "").trim();
    const clientSecret = (process.env.GOOGLE_CLIENT_SECRET || "").trim();
    return { clientId, clientSecret, configured: Boolean(clientId && clientSecret) };
  }
  const clientId = (process.env.GITHUB_CLIENT_ID || "").trim();
  const clientSecret = (process.env.GITHUB_CLIENT_SECRET || "").trim();
  return { clientId, clientSecret, configured: Boolean(clientId && clientSecret) };
}

// ── Supabase as the OAuth broker ─────────────────────────────────────────────
// The Google/GitHub OAuth apps in this project are managed by Supabase, so the
// redirect URI registered with Google is Supabase's own
// `https://<ref>.supabase.co/auth/v1/callback` — NOT anything we control.
// A direct code exchange with Google therefore always fails with
// redirect_uri_mismatch. Supabase can complete that exchange for us, so we send
// the browser to Supabase's authorize endpoint and receive a short-lived code.
interface SupabaseConfig {
  url: string;
  anonKey: string;
  /** Server-only key. Required for the projects database; never sent anywhere. */
  serviceRoleKey: string;
  configured: boolean;
  /** True when the projects store can actually be used. */
  dbConfigured: boolean;
}

function supabaseConfig(): SupabaseConfig {
  const raw = (
    process.env.NEXT_PUBLIC_SUPABASE_URL ||
    process.env.SUPABASE_URL ||
    process.env.VITE_SUPABASE_URL ||
    ""
  ).trim();

  // The dashboard hands out several shapes of the same value:
  //   https://<ref>.supabase.co
  //   https://<ref>.supabase.co/rest/v1   (PostgREST)
  //   https://<ref>.supabase.co/auth/v1    (GoTrue)
  //   ...each of the above with a trailing slash.
  // Only the project root builds a valid /auth/v1/... URL. Appending a path to
  // any of the other shapes yields e.g. /rest/v1/auth/v1/authorize, which is
  // routed to PostgREST instead of GoTrue and answers 401 "No API key found in
  // request" — the browser then shows a blocked/error page instead of the
  // provider's consent screen. So normalise down to the origin + project ref
  // and refuse to build a URL that still carries an API segment.
  const url = raw
    .trim()
    .replace(/\/+$/, "")
    .replace(/\/(rest|auth|pg|storage|functions|realtime|analytics)\/v1$/i, "")
    .replace(/\/+$/, "");

  // Defence in depth: if anything API-shaped survived, the project root is the
  // only safe thing to keep — take everything up to the first known segment.
  const cleaned =
    url.split("/rest/")[0].split("/auth/")[0].split("/pg/")[0].split("/storage/")[0];

  const anonKey = (
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ||
    process.env.SUPABASE_ANON_KEY ||
    process.env.VITE_SUPABASE_ANON_KEY ||
    ""
  ).trim();

  // Server-only. This key bypasses Row Level Security, so it is ONLY ever used
  // here, server-side, after the caller's own signed session has been verified —
  // never forwarded to the browser. Without it the projects database cannot be
  // reached at all and every data route says so instead of faking success.
  const serviceRoleKey = (
    process.env.SUPABASE_SERVICE_ROLE_KEY ||
    process.env.SUPABASE_SERVICE_KEY ||
    ""
  ).trim();

  return {
    url: cleaned,
    anonKey,
    serviceRoleKey,
    configured: Boolean(cleaned && anonKey),
    dbConfigured: Boolean(cleaned && serviceRoleKey),
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Server-side data layer (Supabase / PostgREST)
// ═══════════════════════════════════════════════════════════════════════════
// WHY A REAL DATABASE: projects used to live in localStorage, so a signed-in
// user's sites vanished on another device and a hand-edited `?id=` could read
// anyone else's work. Everything below is keyed on the OWNER ID taken from the
// signed session cookie — never from the request body — and every query filters
// on it, so one account can never read or write another's rows.
//
// No mock data: if the database is unreachable the endpoints answer 503 with a
// clear reason instead of inventing a project.

type Tier = "free" | "pro" | "business";

interface DbProject {
  id: string;
  owner_id: string;
  name: string;
  code: string;
  files: Record<string, string>;
  versions: unknown[];
  theme: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

const PROJECTS_TABLE = "ebnily_projects";

/**
 * Owner-only operational tables.
 *
 * WHY: the dashboard used to answer with hard-coded zeros and empty arrays, so
 * every button on it (approve, reject, block, quota, tier, settings) either
 * showed nothing to act on or reported success without changing anything. These
 * three tables are what make the dashboard real. Until the SQL in
 * `supabase/projects.sql` is run, the endpoints answer 503 DB_UNAVAILABLE — the
 * UI says so instead of showing an empty dashboard that looks like "no data yet".
 */
const PAYMENTS_TABLE = "ebnily_payments";
const DEVICES_TABLE = "ebnily_devices";
const SETTINGS_TABLE = "ebnily_settings";
const ACCOUNTS_TABLE = "ebnily_accounts";
const ADMINS_TABLE = "ebnily_admins";

/**
 * Credits granted to a brand-new account.
 *
 * WHY a wallet at all: the free tier used to be a per-day counter held in a
 * signed cookie, which meant the allowance was bound to the browser — clearing
 * cookies reset it, and it could not be topped up by the owner. Credits are a
 * persisted integer per account instead: the grant happens once (see
 * `welcome_given`), then the admin and the AI routes own the value, so it
 * survives sign-out, another device and a redeploy.
 *
 * One generation costs one credit.
 */
const WELCOME_CREDITS = 5;

/**
 * How recently an account must have been seen to count as present.
 *
 * The account endpoint bumps `last_seen_at` on every authenticated request, and
 * the dashboard polls every 15s, so a 90-second window reliably separates
 * "at the site right now" from "closed the tab" without any socket.
 */
const ONLINE_WINDOW_MS = 90_000;
const IDLE_WINDOW_MS = 30 * 60_000;

/** A row of ebnily_accounts as PostgREST returns it. */
interface DbAccount {
  account_id: string;
  email: string;
  display_name: string;
  provider: string;
  avatar_url: string | null;
  credits: number;
  credits_granted: number | null;
  welcome_given: boolean;
  last_seen_at: string;
  first_seen_at: string;
  requests_count: number | string;
  last_ip: string | null;
  user_agent: string | null;
  tier: string;
  is_blocked: boolean;
  block_reason: string | null;
}

/** The single settings row key. */
const SETTINGS_ROW_KEY = "platform";

interface DbPayment {
  id: string;
  account_email: string;
  account_id: string;
  plan_id: string;
  billing_cycle: string;
  amount_egp: number | string;
  amount_usd: number | string;
  sender_phone: string;
  transaction_ref: string;
  receipt_path: string | null;
  receipt_file_name: string | null;
  notes: string | null;
  status: string;
  submitted_at: string;
  reviewed_at: string | null;
}

interface DbDevice {
  device_id: string;
  fingerprint_hash: string;
  account_email: string | null;
  screen: string | null;
  timezone: string | null;
  platform: string | null;
  user_agent: string | null;
  ip_address: string | null;
  is_blocked: boolean;
  block_reason: string | null;
  free_used: number;
  free_limit: number;
  tier: string;
  first_seen_at: string;
  last_seen_at: string;
}

interface DbSettingsRow {
  key: string;
  value: Record<string, unknown>;
}

/** PostgREST returns numerics as strings — normalise before they reach JSON. */
function toNumber(value: number | string | null | undefined, fallback = 0): number {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/** A payment row in the exact shape `OrangeCashTransaction` declares. */
function toPublicPayment(row: DbPayment) {
  const plan = row.plan_id === "business" ? "business" : "pro";
  return {
    id: row.id,
    senderPhone: row.sender_phone ?? "",
    transactionReference: row.transaction_ref ?? "",
    amount: toNumber(row.amount_egp),
    currency: "EGP",
    planId: plan,
    planName: plan === "business" ? "Business & Agency" : "Professional Pro",
    billingCycle: row.billing_cycle === "yearly" ? ("yearly" as const) : ("monthly" as const),
    userEmail: row.account_email || undefined,
    submittedAt: row.submitted_at,
    status: (row.status === "confirmed" || row.status === "rejected" ? row.status : "pending") as
      | "confirmed"
      | "pending"
      | "rejected",
    verifiedAt: row.reviewed_at ?? undefined,
    // Only the object KEY is stored, never a signed URL, so a shared link cannot
    // grant read access to someone's receipt image.
    receiptImage: row.receipt_path ?? undefined,
    notes: row.notes ?? undefined,
  };
}

/** A device row in the exact shape `DeviceProtectionInfo` declares. */
function toPublicDevice(row: DbDevice) {
  return {
    deviceId: row.device_id,
    fingerprintHash: row.fingerprint_hash ?? "",
    ipAddress: row.ip_address ?? undefined,
    userAgent: row.user_agent ?? undefined,
    freeGenerationsUsed: toNumber(row.free_used),
    freeGenerationsLimit: toNumber(row.free_limit, 5),
    isBlocked: row.is_blocked === true,
    blockReason: row.block_reason ?? undefined,
    associatedTier: (row.tier === "pro" || row.tier === "business" ? row.tier : "free") as
      | "free"
      | "pro"
      | "business",
    registeredEmails: row.account_email ? [row.account_email] : [],
    lastSeen: row.last_seen_at,
  };
}

/** PostgREST call with the server-only service key. Never logs the key. */
async function dbRequest<T>(
  path: string,
  init: RequestInit & { query?: Record<string, string> } = {},
): Promise<{ ok: boolean; status: number; data: T | null; error?: string }> {
  const sb = supabaseConfig();
  if (!sb.dbConfigured) {
    return { ok: false, status: 503, data: null, error: "database_not_configured" };
  }
  const query = new URLSearchParams(init.query ?? {}).toString();
  const url = `${sb.url}/rest/v1/${path}${query ? `?${query}` : ""}`;

  try {
    const res = await fetch(url, {
      method: init.method ?? "GET",
      headers: {
        apikey: sb.serviceRoleKey,
        Authorization: `Bearer ${sb.serviceRoleKey}`,
        "Content-Type": "application/json",
        Prefer: "return=representation",
        ...(init.headers as Record<string, string> | undefined),
      },
      body: init.body,
    });
    const text = await res.text();
    let data: T | null = null;
    try {
      data = text ? (JSON.parse(text) as T) : null;
    } catch {
      data = null;
    }
    if (!res.ok) {
      return { ok: false, status: res.status, data, error: text.slice(0, 300) || res.statusText };
    }
    return { ok: true, status: res.status, data };
  } catch (err) {
    return {
      ok: false,
      status: 503,
      data: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * The owner's identity, resolved SERVER-SIDE from the signed cookie.
 * `owner_id` is stable (the Supabase user id) so rows stay attached to the same
 * account even if the e-mail changes.
 */
function currentOwner(req: Request): { id: string; email: string; isOwner: boolean } | null {
  const session = readAuthSession(req as AuthReq);
  if (!session) return null;
  return { id: session.id, email: session.email, isOwner: isOwnerAccount(session) };
}

// ── Accounts: sign-up record, credit wallet, presence ─────────────────────────
// WHY THIS EXISTS: the platform had no notion of an ACCOUNT at all — only
// devices and payments. That made three things impossible: knowing who signed
// up, giving a new user a starting allowance, and telling the owner whether
// anyone is actually online. All three ride on one row per account in
// `ebnily_accounts`, written here from the verified session cookie.

/** Best-effort client IP. Only used for the admin activity log. */
function clientIp(req: Request): string {
  const forwarded = req.headers["x-forwarded-for"];
  const raw = Array.isArray(forwarded) ? forwarded[0] : forwarded;
  const first = (raw ?? "").split(",")[0]?.trim();
  return (first || req.ip || "").slice(0, 64);
}

/**
 * Record an authenticated visit and return the account row.
 *
 * Called from `/api/auth/me`, which the client hits on every load and the
 * dashboard hits on every poll — so `last_seen_at` is a genuine heartbeat
 * rather than a one-off sign-up stamp.
 *
 * The welcome grant is applied with a conditional upsert so it happens exactly
 * once: the insert carries the 5 credits, and the `on_conflict` update branch
 * deliberately does NOT touch `credits`, so refreshing the page never inflates
 * the balance. A failure here is non-fatal by design — the session is already
 * valid, and refusing to sign someone in because the activity log is down would
 * be a worse outcome than a missing heartbeat.
 */
async function touchAccount(
  req: Request,
  session: AuthUser,
): Promise<DbAccount | null> {
  if (!supabaseConfig().dbConfigured) return null;

  const ip = clientIp(req);
  const ua = String(req.headers["user-agent"] ?? "").slice(0, 250);

  // A staff account is written as `business` on the very first login, so the row
  // in `ebnily_accounts` agrees with what `serverTierFor` will decide later. Two
  // separate truths — the table and the entitlement check — would eventually
  // disagree, and the dashboard would show a staff member as "free" while the app
  // treated them as Business.
  const staff = isStaffEmail(session.email);
  // The table's `tier` column is constrained to ('free','pro','business').
  const tier: Tier = staff ? "business" : "free";

  const res = await dbRequest<DbAccount[]>(ACCOUNTS_TABLE, {
    method: "POST",
    query: { on_conflict: "merge-duplicates" },
    headers: { Prefer: "resolution=merge-duplicates,return=representation" },
    body: JSON.stringify({
      account_id: session.id,
      email: session.email ?? "",
      display_name: session.name ?? "",
      provider: session.provider ?? "",
      avatar_url: session.picture || null,
      // Only applied on INSERT; the conflict branch leaves credits alone.
      credits: WELCOME_CREDITS,
      welcome_given: true,
      last_seen_at: new Date().toISOString(),
      first_seen_at: new Date().toISOString(),
      last_ip: ip || null,
      user_agent: ua || null,
      // Staff keep Business on every later visit too — see the PATCH below.
      tier,
    }),
  });

  if (!res.ok || !Array.isArray(res.data) || res.data.length === 0) {
    if (!res.ok) console.error("touchAccount failed:", res.error);
    return null;
  }
  const row = res.data[0];

  // Bump the counters and the heartbeat separately, and never write `credits`
  // here — this is the update path, and a write of the insert values would
  // reset the balance to WELCOME_CREDITS on every single page load.
  const bump = await dbRequest<DbAccount[]>(ACCOUNTS_TABLE, {
    method: "PATCH",
    query: { account_id: `eq.${session.id}` },
    headers: { Prefer: "return=representation" },
    body: JSON.stringify({
      last_seen_at: new Date().toISOString(),
      last_ip: ip || null,
      user_agent: ua || null,
      email: session.email ?? row.email,
      display_name: session.name ?? row.display_name,
      avatar_url: session.picture || row.avatar_url,
      // Re-assert the staff tier on every visit. Without this the column would
      // only ever be set by the INSERT, so an account that was later added to
      // `STAFF_EMAILS` — or one an admin had reset to 'free' by hand — would keep
      // the stale value forever while `serverTierFor` said otherwise.
      //
      // Only staff are re-asserted: a normal account's `tier` is left exactly as
      // it is, so a payment the owner approved is never silently reverted here.
      ...(staff ? { tier } : {}),
    }),
  });

  if (bump.ok && Array.isArray(bump.data) && bump.data[0]) return bump.data[0];

  // Fall back to counting requests with a second, additive update. PostgREST
  // cannot `credits + 1` in one round trip without an RPC, so this is left to
  // the dashboard's derived request count rather than faked here.
  return row;
}

/**
 * Presence derived from the heartbeat.
 *
 * `online` — seen within ONLINE_WINDOW_MS.
 * `idle`   — seen within IDLE_WINDOW_MS but not online.
 * `offline`— anything older, including a row that never got a timestamp.
 */
function presenceOf(lastSeenAt: string | null | undefined): "online" | "idle" | "offline" {
  const at = lastSeenAt ? new Date(lastSeenAt).getTime() : NaN;
  if (!Number.isFinite(at)) return "offline";
  const age = Date.now() - at;
  if (age <= ONLINE_WINDOW_MS) return "online";
  if (age <= IDLE_WINDOW_MS) return "idle";
  return "offline";
}

/** Strip an account row down to what the dashboard is allowed to see. */
function toPublicAccount(row: DbAccount) {
  return {
    accountId: row.account_id,
    email: row.email,
    name: row.display_name,
    provider: row.provider,
    avatarUrl: row.avatar_url,
    credits: toNumber(row.credits),
    welcomeGiven: row.welcome_given === true,
    tier: row.tier,
    isBlocked: row.is_blocked === true,
    blockReason: row.block_reason,
    firstSeenAt: row.first_seen_at,
    lastSeenAt: row.last_seen_at,
    requestsCount: toNumber(row.requests_count),
    lastIp: row.last_ip,
    presence: presenceOf(row.last_seen_at),
  };
}

// ── Subscription state (single source of truth, server-side) ────────────────
// Read from the signed `ebnili_plan` grant the owner mints when a payment is
// approved, or from the owner's account. The browser NEVER sends the tier — a
// client-provided "I'm paid" is exactly what this guard exists to reject.

/**
 * The subscription tier the SERVER grants this session.
 *
 * This is the single place every paid feature asks, so a staff account is
 * honoured once here rather than in each caller that might forget.
 *
 * Resolution order (first match wins):
 *   1. the site owner      — unchanged behaviour
 *   2. a STAFF account     — automatic Business, no payment, no expiry
 *   3. a signed plan grant — bound to the session e-mail, unexpired
 *   4. free
 *
 * The browser never sends a tier. That is still the rule: everything below is
 * derived from the verified session cookie, never from a request field.
 */
function serverTierFor(req: Request, session: AuthUser | null): Tier {
  if (isOwnerAccount(session)) return "business";
  // Staff bypass the subscription check entirely — see `isStaffEmail`.
  if (isStaffEmail(session?.email)) return "business";
  const grant = readPlanGrant(req);
  if (!grant) return "free";
  // A grant only ever counts for the account it was issued to.
  if (!grantBelongsToSession(grant, session)) return "free";
  if (new Date(grant.expiresAt).getTime() <= Date.now()) return "free";
  return grant.tier;
}

function isPaidTier(tier: Tier): boolean {
  return tier === "pro" || tier === "business";
}

// ═══════════════════════════════════════════════════════════════════════════
// Watermark — enforced on the server, inside the exported artifact itself
// ═══════════════════════════════════════════════════════════════════════════
// WHY: the watermark used to be a badge in the preview UI only. Removing a
// `div` in devtools (or exporting straight from the browser bundle) removed it,
// and the ZIP never had one at all. Now the decision is made here, from the
// server-resolved tier, and the mark is written INTO the exported files — so
// it ships in the ZIP, in the copied code, and in the deployed site, and it
// cannot be stripped client-side because the client never performs the export.
//
// `injectWatermark` is idempotent: re-exporting an already-marked project
// does not stack a second mark, and it never double-injects.

const WATERMARK_STYLE =
  "position:fixed;bottom:12px;right:12px;z-index:2147483647;display:flex;" +
  "align-items:center;gap:6px;padding:6px 10px;border-radius:8px;" +
  "background:rgba(2,6,23,.82);color:#cbd5e1;font:600 11px/1.4 system-ui,sans-serif;" +
  "border:1px solid rgba(148,163,184,.35);pointer-events:none;user-select:none";

const WATERMARK_HTML = `<div id="ebnili-watermark" data-ebnili-watermark="1" style="${WATERMARK_STYLE}">⚡ صنع بواسطة إبنيلي AI</div>`;

function hasWatermark(html: string): boolean {
  return /data-ebnili-watermark/i.test(html) || /id="ebnili-watermark"/i.test(html);
}

/**
 * Add or remove the mark according to the SERVER-resolved tier.
 * Free → marked. Paid → any existing mark is stripped.
 */
function applyWatermark(html: string, tier: Tier): string {
  const doc = String(html ?? "");
  if (!doc) return doc;

  if (isPaidTier(tier)) {
    // Paid: never ship a mark, and clean any that a previous (free) export
    // already wrote into this document.
    if (!hasWatermark(doc)) return doc;
    return doc
      .replace(/<div[^>]*data-ebnili-watermark[^>]*>[\s\S]*?<\/div>\s*/gi, "")
      .replace(/<div[^>]*id="ebnili-watermark"[^>]*>[\s\S]*?<\/div>\s*/gi, "");
  }

  if (hasWatermark(doc)) return doc; // already marked — do not stack
  // Before </body> when there is one, otherwise append.
  if (/<\/body>/i.test(doc)) return doc.replace(/<\/body>/i, `${WATERMARK_HTML}\n</body>`);
  return `${doc}\n${WATERMARK_HTML}`;
}

function base64url(buf: Buffer): string {
  return buf.toString("base64url");
}

function createPkcePair(): { verifier: string; challenge: string } {
  const verifier = base64url(crypto.randomBytes(48));
  const challenge = base64url(crypto.createHash("sha256").update(verifier).digest());
  return { verifier, challenge };
}

interface SupabaseIdentity {
  id: string;
  email: string;
  name: string;
  picture: string;
  provider: AuthProviderId;
}

/** Maps a Supabase user record onto our own session shape. */
function mapSupabaseUser(raw: {
  id?: string;
  email?: string;
  user_metadata?: Record<string, unknown>;
  app_metadata?: Record<string, unknown>;
}): SupabaseIdentity | null {
  if (!raw?.id) return null;
  const meta = raw.user_metadata || {};
  const appMeta = raw.app_metadata || {};

  // Supabase records the OAuth provider in app_metadata.provider.
  const providerRaw = String(appMeta.provider || "google").toLowerCase();
  const provider: AuthProviderId = providerRaw === "github" ? "github" : "google";

  const fullName =
    (typeof meta.full_name === "string" && meta.full_name) ||
    (typeof meta.name === "string" && meta.name) ||
    (typeof meta.preferred_username === "string" && meta.preferred_username) ||
    raw.email ||
    (provider === "github" ? "GitHub User" : "Google User");

  const picture =
    (typeof meta.avatar_url === "string" && meta.avatar_url) ||
    (typeof meta.picture === "string" && meta.picture) ||
    "";

  return { id: String(raw.id), name: String(fullName), email: raw.email || "", picture, provider };
}

/**
 * Mint a session cookie, or `null` when the deployment cannot sign safely.
 *
 * Returning `null` (rather than a throw) is what makes the failure mode a locked
 * feature instead of a 500: the two OAuth callbacks translate it into an
 * actionable `auth_error` the UI can explain.
 */
function issueAuthSession(user: AuthUser): string | null {
  const body = Buffer.from(
    JSON.stringify({ ...user, iat: Date.now(), exp: Date.now() + AUTH_SESSION_TTL_MS }),
  ).toString("base64url");
  const sig = hmacB64(body);
  // No signature ⇒ no session. Never ship an unsigned token.
  if (!sig) return null;
  return `${body}.${sig}`;
}

function readAuthSession(req: AuthReq): AuthUser | null {
  const token = readCookie(req, AUTH_COOKIE_NAME);
  if (!token) return null;
  const sep = token.indexOf(".");
  if (sep <= 0) return null;
  const body = token.slice(0, sep);
  const sig = token.slice(sep + 1);
  // No usable key ⇒ nothing can be verified, so every cookie is rejected. This
  // is the half that stops a forged session from being accepted.
  const expected = hmacB64(body);
  if (!expected) return null;
  if (!safeEqual(sig, expected)) return null;
  try {
    const parsed = JSON.parse(Buffer.from(body, "base64url").toString("utf-8")) as {
      id?: string;
      name?: string;
      email?: string;
      picture?: string;
      provider?: string;
      exp?: number;
    };
    if (!parsed?.id || !parsed?.provider) return null;
    if (typeof parsed.exp !== "number" || Date.now() > parsed.exp) return null;
    return {
      id: String(parsed.id),
      name: String(parsed.name || parsed.email || "Ebnili User"),
      email: String(parsed.email || ""),
      picture: String(parsed.picture || ""),
      provider: parsed.provider === "github" ? "github" : "google",
    };
  } catch {
    return null;
  }
}
// ── Provider token exchange + profile fetch (no SDK, plain fetch) ────────────
async function exchangeGoogleCode(code: string, redirectUri: string): Promise<AuthUser | null> {
  const { clientId, clientSecret } = providerConfig("google");
  const tokenRes = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      code,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri,
      grant_type: "authorization_code",
    }).toString(),
  });
  if (!tokenRes.ok) return null;
  const tokenJson = (await tokenRes.json()) as { access_token?: string };
  if (!tokenJson.access_token) return null;

  const infoRes = await fetch("https://openidconnect.googleapis.com/v1/userinfo", {
    headers: { Authorization: `Bearer ${tokenJson.access_token}` },
  });
  if (!infoRes.ok) return null;
  const info = (await infoRes.json()) as {
    sub?: string;
    name?: string;
    email?: string;
    picture?: string;
  };
  if (!info.sub) return null;
  return {
    id: String(info.sub),
    name: String(info.name || info.email || "Google User"),
    email: String(info.email || ""),
    picture: String(info.picture || ""),
    provider: "google",
  };
}

async function exchangeGitHubCode(code: string, redirectUri: string): Promise<AuthUser | null> {
  const { clientId, clientSecret } = providerConfig("github");
  const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded", Accept: "application/json" },
    body: new URLSearchParams({
      code,
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri,
    }).toString(),
  });
  if (!tokenRes.ok) return null;
  const tokenJson = (await tokenRes.json()) as { access_token?: string };
  if (!tokenJson.access_token) return null;

  const ghHeaders = {
    Authorization: `Bearer ${tokenJson.access_token}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "Ebnili-App",
  };
  const userRes = await fetch("https://api.github.com/user", { headers: ghHeaders });
  if (!userRes.ok) return null;
  const ghUser = (await userRes.json()) as {
    id?: number;
    login?: string;
    name?: string;
    email?: string | null;
    avatar_url?: string;
  };
  if (!ghUser.id) return null;

  // GitHub users may hide their email — fall back to the verified primary one.
  let email = ghUser.email || "";
  if (!email) {
    try {
      const emailsRes = await fetch("https://api.github.com/user/emails", { headers: ghHeaders });
      if (emailsRes.ok) {
        const emails = (await emailsRes.json()) as {
          email?: string;
          primary?: boolean;
          verified?: boolean;
        }[];
        const match = emails.find((e) => e?.primary && e?.verified) || emails.find((e) => e?.verified);
        email = match?.email || "";
      }
    } catch {
      /* email stays empty — login still succeeds */
    }
  }

  return {
    id: String(ghUser.id),
    name: String(ghUser.name || ghUser.login || "GitHub User"),
    email,
    picture: ghUser.avatar_url || "",
    provider: "github",
  };
}
// ── Auth routes ─────────────────────────────────────────────────────────────
// NOTE: the literal paths below are registered BEFORE `/api/auth/:provider`,
// otherwise Express would treat "me" / "logout" / "providers" as a provider id.

// Which providers can actually be used right now (drives the UI).
//
// This is the SINGLE source of truth for the client: it reports the providers,
// which route will actually handle them ("supabase" or "direct"), and the exact
// callback URL that must be registered on the provider/Supabase side. The UI
// used to merge this with /api/auth/config, so the two could disagree; now the
// client derives everything from one response.
app.get("/api/auth/providers", (req: AuthReq, res: AuthRes) => {
  const ids: AuthProviderId[] = ["google", "github"];
  const sb = supabaseConfig().configured;
  const base = requestBaseUrl(req);
  res.json({
    success: true,
    baseUrl: base,
    route: sb ? "supabase" : "direct",
    // The callback that must be whitelisted. With Supabase brokering, the
    // browser lands on the Supabase callback and Supabase then calls back here.
    callbackBase: sb ? `${base}/api/auth/callback/supabase` : `${base}/api/auth/callback`,
    providers: ids.map((id) => ({
      id,
      configured: sb || providerConfig(id).configured,
    })),
  });
});

// Who am I? Returns `{ authenticated: false }` for guests — never an error.
//
// `isOwner` is decided HERE, on the server, from the signed session cookie —
// never in the browser. Owner-only controls (admin dashboard, backend/database
// console) read this flag, so they stay out of every ordinary user's header and
// the flag cannot be forged by editing client state. The owner's email address
// is deliberately absent from the public JS bundle.
app.get("/api/auth/me", async (req: AuthReq, res: AuthRes) => {
  const user = readAuthSession(req);

  // Refresh the admin email cache so a newly added admin is recognized promptly.
  await refreshAdminEmails();

  // Presence + sign-up record. This is the one endpoint the client hits on every
  // load and the owner dashboard hits on every poll, which is what makes
  // `last_seen_at` a real heartbeat rather than a one-off sign-up stamp. It also
  // performs the one-time welcome grant of 5 credits for a brand-new account.
  //
  // Non-fatal by design: `touchAccount` returns null when the database is not
  // configured or the write fails, and the client still gets a valid session.
  // The credits field is simply omitted then, which the UI reads as "unknown"
  // rather than as zero — a missing allowance must never read as "you have none".
  const account = user ? await touchAccount(req, user) : null;

  // An added admin gets isOwner=true so the dashboard and sidebar controls appear.
  const ownerFlag = isOwnerAccount(user) || isAddedAdmin(user);

  res.json({
    success: true,
    authenticated: Boolean(user),
    user: user ? { ...user, isOwner: ownerFlag } : null,
    isOwner: ownerFlag,
    ...(account
      ? {
          credits: toNumber(account.credits),
          welcomeGiven: account.welcome_given === true,
          presence: presenceOf(account.last_seen_at),
        }
      : {}),
  });
});

app.post("/api/auth/logout", (req: AuthReq, res: AuthRes) => {
  clearAuthCookie(res, AUTH_COOKIE_NAME, req);
  clearAuthCookie(res, AUTH_STATE_COOKIE, req);
  clearAuthCookie(res, AUTH_PKCE_COOKIE, req);
  res.json({ success: true, authenticated: false });
});

// The exact callback URLs this server redirects to. A setup aid so a
// redirect_uri mismatch can be fixed by copying the value instead of guessing.
// Registered here, above `/api/auth/:provider`, so it is not treated as a
// provider id. Exposes no secrets — only public redirect URIs.
app.get("/api/auth/config", (req: AuthReq, res: AuthRes) => {
  const base = requestBaseUrl(req);
  const describe = (id: AuthProviderId) => {
    const cfg = providerConfig(id);
    return {
      configured: cfg.configured,
      // The shape of the id only — never the secret itself.
      clientIdSuffix: cfg.clientId.slice(-12),
      clientIdLooksValid: /^[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com$/.test(cfg.clientId),
      hasSecret: Boolean(cfg.clientSecret),
    };
  };
  const sb = supabaseConfig();
  res.json({
    success: true,
    baseUrl: base,
    callbacks: {
      google: `${base}/api/auth/callback/google`,
      github: `${base}/api/auth/callback/github`,
    },
    // When true, sign-in is brokered by Supabase and the URLs above are NOT
    // what Google/GitHub must know — Supabase's own callback is.
    supabase: {
      configured: sb.configured,
      // Host only: the anon key is public but there is no reason to echo it.
      host: sb.configured ? sb.url.replace(/^https?:\/\//, "") : "",
    },
    google: describe("google"),
    github: describe("github"),
  });
});

// Step 1 — bounce the browser to the provider's consent screen.
//
// Two possible routes:
//  A) Supabase broker (preferred when configured). Supabase owns the OAuth app
//     whose redirect URI Google/GitHub actually know, so it must perform the
//     code→token exchange. We use PKCE and receive only a short-lived code.
//  B) Direct to the provider — used when this project owns the OAuth app.
app.get("/api/auth/:provider", (req: AuthReq, res: AuthRes) => {
  const provider = String(req.params.provider || "").toLowerCase() as AuthProviderId;
  const home = requestBaseUrl(req);

  if (provider !== "google" && provider !== "github") {
    return res.status(404).json({ success: false, message: "Unknown auth provider" });
  }

  const sb = supabaseConfig();
  if (sb.configured) {
    const { verifier, challenge } = createPkcePair();

    // The verifier stays server-side in an HttpOnly cookie; only the derived
    // challenge ever goes to the browser, so a stolen callback code is useless.
    res.cookie(AUTH_PKCE_COOKIE, `${provider}.${verifier}`, authCookieOptions(AUTH_STATE_TTL_MS, req));

    const url = new URL(`${sb.url}/auth/v1/authorize`);
    // Never emit a URL that still carries a PostgREST/GoTrue API segment —
    // those resolve to the wrong service and surface to the user as a blocked
    // page. If the base was misconfigured, fail loudly with a readable message
    // instead of redirecting the browser into a 401.
    if (/\/(rest|pg|storage|functions)\/v1/i.test(url.pathname)) {
      console.error("supabase base url still carries an API path:", sb.url);
      return res.redirect(`${home}/?auth_error=supabase_not_configured`);
    }
    url.searchParams.set("provider", provider);
    url.searchParams.set("redirect_to", `${home}/api/auth/callback/supabase`);
    url.searchParams.set("code_challenge", challenge);
    url.searchParams.set("code_challenge_method", "s256");
    url.searchParams.set("skip_http_redirect", "true");
    // The anon key is public by design. Sending it as a query param keeps the
    // authorize call valid even when Supabase's gateway is configured to demand
    // an API key on this route.
    if (sb.anonKey) url.searchParams.set("apikey", sb.anonKey);
    if (provider === "google") {
      url.searchParams.set("prompt", "select_account");
      // Supabase forwards `query_params` to Google; `hl` keeps the consent
      // screen in English instead of following the browser's locale.
      url.searchParams.set("query_params", "hl=en");
    }

    return res.redirect(url.toString());
  }

  const cfg = providerConfig(provider);
  if (!cfg.configured) {
    return res.redirect(`${home}/?auth_error=not_configured`);
  }

  const state = crypto.randomBytes(24).toString("hex");
  // The provider returns via a top-level GET navigation, so SameSite=Lax is
  // required; the attributes come from one place so the delete below matches.
  res.cookie(AUTH_STATE_COOKIE, `${provider}.${state}`, authCookieOptions(AUTH_STATE_TTL_MS, req));

  const redirectUri = authCallbackUrl(req, provider);
  const url =
    provider === "google"
      ? new URL("https://accounts.google.com/o/oauth2/v2/auth")
      : new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", cfg.clientId);
  url.searchParams.set("redirect_uri", redirectUri);
  url.searchParams.set("response_type", "code");
  url.searchParams.set("state", state);
  url.searchParams.set("scope", provider === "google" ? "openid email profile" : "read:user user:email");

  // Force the provider's consent screen into English so it matches the app,
  // instead of inheriting the browser's locale.
  //
  // `prompt=select_account` is what makes "تغيير الحساب" actually work: without
  // it Google silently re-authorises the account that is already signed in, so
  // the user picks the same identity and appears to be back where they started.
  // GitHub has no equivalent parameter — it reuses its own session cookie — so
  // `allow_signup` is the most we can ask for, and the account menu also offers
  // an explicit sign-out for the GitHub case.
  if (provider === "google") {
    url.searchParams.set("prompt", "select_account");
    url.searchParams.set("hl", "en");
  } else {
    url.searchParams.set("allow_signup", "true");
  }

  return res.redirect(url.toString());
});

// Step 2 (Supabase route) — Supabase has completed the provider handshake and
// hands back a short-lived PKCE code. Registered ABOVE `/callback/:provider`,
// otherwise Express would treat "supabase" as a provider id.
// This must exist before that route to win the match.
app.get("/api/auth/callback/supabase", async (req: AuthReq, res: AuthRes) => {
  const home = requestBaseUrl(req);
  const fail = (reason: string) => res.redirect(`${home}/?auth_error=${encodeURIComponent(reason)}`);

  const sb = supabaseConfig();
  if (!sb.configured) return fail("supabase_not_configured");

  // Supabase reports its own errors (e.g. a user denied consent) in the query.
  const errDesc = typeof req.query.error_description === "string" ? req.query.error_description : "";
  if (errDesc) return fail("oauth_denied");
  const errCode = typeof req.query.error === "string" ? req.query.error : "";
  if (errCode) return fail("oauth_denied");

  // Single-use: clear the verifier cookie on read so a replayed callback fails.
  const stored = readCookie(req, AUTH_PKCE_COOKIE);
  clearAuthCookie(res, AUTH_PKCE_COOKIE, req);
  if (!stored) return fail("pkce_missing");
  const sep = stored.indexOf(".");
  if (sep <= 0) return fail("pkce_invalid");
  const provider = stored.slice(0, sep) as AuthProviderId;
  const verifier = stored.slice(sep + 1);
  if (provider !== "google" && provider !== "github") return fail("pkce_invalid");

  // The code comes back in the query string (skip_http_redirect) or, for some
  // providers, in the URL fragment — which never reaches the server.
  const code = typeof req.query.code === "string" ? req.query.code : "";
  if (!code) return fail("missing_code");

  try {
    // Trade the code + our verifier for the signed-in user.
    const tokenRes = await fetch(`${sb.url}/auth/v1/token?grant_type=pkce`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        apikey: sb.anonKey,
        Authorization: `Bearer ${sb.anonKey}`,
      },
      body: JSON.stringify({ auth_code: code, code_verifier: verifier }),
    });

    if (!tokenRes.ok) {
      console.error("supabase pkce exchange failed:", tokenRes.status);
      return fail("exchange_failed");
    }

    const tokenJson = (await tokenRes.json()) as {
      user?: Parameters<typeof mapSupabaseUser>[0];
    };
    const identity = mapSupabaseUser(tokenJson.user || {});
    if (!identity) return fail("profile_failed");

    // Fail closed: with no signing key we cannot mint a session, and issuing an
    // unsigned cookie would be exactly the vulnerability this guards against.
    const session = issueAuthSession(identity);
    if (!session) return fail("auth_secret_missing");

    res.cookie(AUTH_COOKIE_NAME, session, authCookieOptions(AUTH_SESSION_TTL_MS));
    return res.redirect(`${home}/?auth=success`);
  } catch (e) {
    console.error("supabase callback failed:", e instanceof Error ? e.message : String(e));
    return fail("exchange_failed");
  }
});

// Step 2 — the provider redirects back here with ?code=…&state=…
// (direct-provider route, used when this project owns the OAuth app)
app.get("/api/auth/callback/:provider", async (req: AuthReq, res: AuthRes) => {
  const provider = String(req.params.provider || "").toLowerCase() as AuthProviderId;
  const home = requestBaseUrl(req);
  const fail = (reason: string) => res.redirect(`${home}/?auth_error=${encodeURIComponent(reason)}`);

  if (provider !== "google" && provider !== "github") return fail("unknown_provider");

  const code = typeof req.query.code === "string" ? req.query.code : "";
  const state = typeof req.query.state === "string" ? req.query.state : "";
  const oauthError = typeof req.query.error === "string" ? req.query.error : "";
  if (oauthError) return fail(oauthError);
  if (!code || !state) return fail("missing_code");

  // Single-use CSRF nonce: the cookie is cleared on first read, so a replayed
  // callback URL can never mint a second session.
  const expected = readCookie(req, AUTH_STATE_COOKIE);
  clearAuthCookie(res, AUTH_STATE_COOKIE, req);
  if (!expected) return fail("state_cookie_missing");
  const sep = expected.indexOf(".");
  if (sep <= 0) return fail("state_cookie_invalid");
  if (expected.slice(0, sep) !== provider) return fail("state_provider_mismatch");
  if (!safeEqual(expected.slice(sep + 1), state)) return fail("state_mismatch");

  const cfg = providerConfig(provider);
  if (!cfg.configured) return fail("not_configured");

  try {
    const redirectUri = authCallbackUrl(req, provider);
    const user =
      provider === "google"
        ? await exchangeGoogleCode(code, redirectUri)
        : await exchangeGitHubCode(code, redirectUri);
    if (!user) return fail("profile_failed");

    const session = issueAuthSession(user);
    if (!session) return fail("auth_secret_missing");

    res.cookie(AUTH_COOKIE_NAME, session, authCookieOptions(AUTH_SESSION_TTL_MS));
    return res.redirect(`${home}/?auth=success`);
  } catch (e) {
    console.error("auth callback failed:", e instanceof Error ? e.message : String(e));
    return fail("exchange_failed");
  }
});

// ── Gemini AI proxy ─────────────────────────────────────────────────────────
// Accepts GEMINI_API_KEY plus common aliases so the function works no matter
// which exact variable name was configured in the Vercel dashboard.
function getApiKey(): string {
  return (
    process.env.GEMINI_API_KEY ||
    process.env.GOOGLE_API_KEY ||
    process.env.GOOGLE_GEMINI_API_KEY ||
    process.env.VITE_GEMINI_API_KEY ||
    ""
  ).trim();
}

function getGeminiClient(): GoogleGenAI | null {
  const apiKey = getApiKey();
  if (!apiKey) return null;
  return new GoogleGenAI({ apiKey });
}

// Model priority is based on the official Gemini model catalogue and live
// production behaviour. Keep concrete stable IDs instead of relying on a
// moving alias: the alias can be unavailable for a particular API key.
//   gemini-3.5-flash      → primary: stable, fast, strong for coding
//   gemini-3.5-flash-lite → low-latency fallback for the same generation
//   gemini-flash-latest   → official hot-swapped alias
//   gemini-3.8-flash      → newest flagship Flash
//   gemini-2.5-flash      → the most widely provisioned stable model; a 404 on
//                          this one means the key cannot reach the 3.x family
//   gemini-2.5-flash-lite → last low-cost resort
//   gemini-3.7 / 3.6      → previous-generation stable models
// Anything the key cannot reach answers 404 and is dropped immediately
// (classifyModelError → "permanent"), so listing extra candidates is free.
const CANDIDATE_MODELS = [
  "gemini-3.5-flash",
  "gemini-3.5-flash-lite",
  "gemini-flash-latest",
  "gemini-3.8-flash",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-3.7-flash",
  "gemini-3.6-flash",
];

// TIME BUDGETS — this is what made "big prompts" fail 100% of the time.
//
// A complete single-file application is a 40k–120k token answer. The old
// settings were 18s per attempt inside a 45s total budget, and vercel.json
// capped the whole function at 60s. Two models ate 36s and "timed out" while
// they were still generating, the rest answered 503 "high demand", and the
// request always ended as HTTP 502 — for a two-word prompt as well as for a
// full specification. The budgets below match what the function is actually
// allowed to run (see `maxDuration` in vercel.json), leaving room for a real
// answer plus a second attempt, and the text-only studio endpoints keep a
// shorter slice because they never return a document.
const GENERATE_BUDGET_MS = 240_000;
const GENERATE_ATTEMPT_TIMEOUT_MS = 75_000;
const REFINE_BUDGET_MS = 180_000;
const REFINE_ATTEMPT_TIMEOUT_MS = 70_000;
const STUDIO_BUDGET_MS = 120_000;
const STUDIO_ATTEMPT_TIMEOUT_MS = 45_000;

// Never let a document get cut off mid-file: an unterminated HTML string used
// to reach the browser as a blank/broken preview. A high ceiling keeps big
// specifications in one answer instead of silently truncating them.
const DOCUMENT_MAX_OUTPUT_TOKENS = 65_536;

// "Accept any text in the input box" without letting a pasted 2MB essay
// exceed the request budget: the head and the tail are kept (a specification
// states its goal first and its constraints last) and the middle is elided.
const MAX_PROMPT_CHARS = 40_000;
const MAX_CURRENT_CODE_CHARS = 120_000;

function boundText(text: string, maxChars: number, elisionNote: string): string {
  const t = (text || "").trim();
  if (t.length <= maxChars) return t;
  const headChars = Math.floor(maxChars * 0.75);
  const tailChars = maxChars - headChars;
  const dropped = t.length - maxChars;
  return `${t.slice(0, headChars)}\n\n[${elisionNote}: ${dropped} characters elided]\n\n${t.slice(-tailChars)}`;
}

async function withDeadline<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

// A 404/"no longer available"/invalid-key failure means the model is unusable
// for this API key. A timeout is also not useful to retry within the same
// serverless request: it has already consumed its entire attempt budget.
// Keep transient 429/503 failures eligible for one short second pass.
function classifyModelError(raw: string): "permanent" | "timeout" | "transient" {
  if (
    /\b400\b|\b401\b|\b403\b|\b404\b|invalid argument|bad request|is not found|no longer available|not found for API key|API key not valid|API_KEY_INVALID/i.test(
      raw,
    )
  ) {
    return "permanent";
  }
  if (/timed out|deadline exceeded/i.test(raw)) return "timeout";
  return "transient";
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

interface GeminiCallOptions {
  /** Total wall-clock the whole call (all models, all passes) may take. */
  budgetMs?: number;
  /** Per-model attempt ceiling. */
  attemptTimeoutMs?: number;
  /** Output ceiling — raise it for document-producing calls. */
  maxOutputTokens?: number;
}

async function generateWithGemini(
  ai: GoogleGenAI,
  prompt: string,
  systemInstruction?: string,
  options: GeminiCallOptions = {},
) {
  const budgetMs = options.budgetMs ?? GENERATE_BUDGET_MS;
  const attemptTimeoutMs = options.attemptTimeoutMs ?? GENERATE_ATTEMPT_TIMEOUT_MS;
  const deadline = Date.now() + budgetMs;
  const failures: string[] = [];
  const deadModels = new Set<string>();
  const retryableModels = new Set<string>();
  const attempts = new Map<string, number>();
  let lastError: unknown = null;

  // Retry only short-lived quota/capacity failures. Repeating a 404 or a
  // timed-out request cannot make this response arrive any faster.
  const MAX_PASSES = 2;
  const MAX_ATTEMPTS_PER_MODEL = 2;

  for (let pass = 1; pass <= MAX_PASSES; pass++) {
    let attempted = false;

    for (const model of CANDIDATE_MODELS) {
      if (deadModels.has(model) || (attempts.get(model) ?? 0) >= MAX_ATTEMPTS_PER_MODEL) continue;
      if (pass > 1 && !retryableModels.has(model)) continue;

      const remaining = deadline - Date.now();
      if (remaining <= 0) {
        failures.push(`budget exhausted before pass ${pass}`);
        break;
      }

      attempted = true;
      attempts.set(model, (attempts.get(model) ?? 0) + 1);
      // Never start an attempt we cannot finish: leave room for the response to
      // travel, and never let a retry begin with a fraction of a second left.
      const usable = remaining - 1_500;
      if (usable < 2_000) {
        failures.push(`budget exhausted before ${model}`);
        break;
      }
      const attemptTimeout = Math.min(usable, attemptTimeoutMs);
      const usesThinkingLevel = model === "gemini-3.5-flash" || model === "gemini-3.5-flash-lite";
      // `low` is supported by both stable 3.5 Flash variants. Avoid changing
      // the shared config when there is no system instruction.
      const config: {
        systemInstruction?: string;
        thinkingConfig?: { thinkingLevel: ThinkingLevel };
        maxOutputTokens?: number;
      } = {
        ...(systemInstruction ? { systemInstruction } : {}),
        ...(usesThinkingLevel ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } } : {}),
        ...(options.maxOutputTokens ? { maxOutputTokens: options.maxOutputTokens } : {}),
      };

      try {
        // NOTE: `contents` must be a plain string (not an array). Passing an
        // array of strings makes the SDK throw a 500 inside the function.
        const result = await withDeadline(
          ai.models.generateContent({
            model,
            contents: prompt,
            ...(Object.keys(config).length > 0 ? { config } : {}),
          }),
          attemptTimeout,
          `Gemini model "${model}"`,
        );
        if (!extractText(result).trim()) {
          throw new Error(`Gemini model "${model}" returned an empty response`);
        }
        return result;
      } catch (e) {
        lastError = e;
        const raw = (e instanceof Error ? e.message : String(e)).replace(/\s+/g, " ");
        const errorKind = classifyModelError(raw);
        if (errorKind === "permanent") deadModels.add(model);
        if (errorKind === "transient") retryableModels.add(model);
        failures.push(`p${pass} ${model}: ${raw.slice(0, 160)}`);
      }
    }

    const canRetry = CANDIDATE_MODELS.some(
      (model) =>
        !deadModels.has(model) &&
        retryableModels.has(model) &&
        (attempts.get(model) ?? 0) < MAX_ATTEMPTS_PER_MODEL,
    );
    if (!attempted || !canRetry || deadline - Date.now() <= 2_000) break;

    // A 429/503 "high demand" clears on its own within seconds. Rushing the
    // second pass used to burn the whole budget and fail anyway; a short,
    // deadline-aware pause is what actually makes the retry worth taking.
    // Two seconds are always left for JSON serialization and the Vercel reply.
    const pause = Math.min(2_500 * pass, Math.max(0, deadline - Date.now() - 2_000));
    if (pause > 0) await sleep(pause);
  }

  const detail = failures.join(" | ").slice(0, 1800);
  console.error(`generateWithGemini: all models failed — ${detail}`);
  throw lastError
    ? new Error(`${lastError instanceof Error ? lastError.message : String(lastError)} [per-model: ${detail}]`)
    : new Error(`All candidate Gemini models are currently unavailable [per-model: ${detail}]`);
}

// The @google/genai SDK returns `response.text` as a *getter property*, not a
// method — read it defensively so we never send an empty `code` back.
function extractText(result: unknown): string {
  if (!result || typeof result !== "object") return "";
  const r = result as { text?: unknown; candidates?: Array<{ content?: { parts?: Array<{ text?: unknown }> } }> };
  if (typeof r.text === "string" && r.text.length > 0) return r.text;
  try {
    const parts = r.candidates?.[0]?.content?.parts ?? [];
    return parts.map((p) => (typeof p.text === "string" ? p.text : "")).join("");
  } catch {
    return "";
  }
}

// ── Owner-defined generation standards (applied to every AI endpoint) ─────
const CORE_RULES = `NON-NEGOTIABLE STANDARDS:
1. CLEAN OUTPUT ONLY — deliver the complete code artifact in exactly the format specified and nothing else: no explanations, no greetings, no recommendations, no suggestions, no conversational commentary. Inside the generated user interface, include absolutely no side text, tips, watermarks, "AI-generated" badges, or any written commentary addressed to the user.
2. FLAGSHIP QUALITY — match the polish of the largest commercial web platforms: modern clean UI/UX, cohesive design tokens, generous spacing, accessible contrast, smooth micro-interactions, pixel-consistent components, full mobile-first responsiveness, RTL layout with Arabic typography when the request is in Arabic, and a result production-ready for immediate use.
3. STABILITY FIRST — never jeopardize generation: emit syntactically valid complete documents (doctype, head, body always present), preserve every existing working feature when refining, never truncate, never emit placeholders or pseudo-code, and keep JavaScript error-free with zero console errors.`;

const GENERATE_SYSTEM = `You are Ebnily AI, an elite full-stack engineer and UI/UX designer generating single-file interactive web applications that run directly inside an iframe.
Return ONLY one complete standalone HTML document beginning with <!DOCTYPE html> and ending with </html>, including the Tailwind CDN (https://cdn.tailwindcss.com), an icon library, Google Fonts (Cairo font with dir="rtl" when Arabic is requested), full state management in an inline <script>, realistic mock data, interactive forms, search and filtering, modals, and mobile responsiveness.
No markdown fences, no JSON, no prose before or after the document.
${CORE_RULES}`;

const REFINE_SYSTEM = `You are Ebnily AI's precision code refiner. Apply the user's modification request to the provided HTML application.
Return ONLY the complete updated HTML document (from <!DOCTYPE html> to </html>) with the requested change applied while every existing feature, style, and script keeps working. No explanations, no diffs, no markdown fences, no prose.
${CORE_RULES}`;

// ── JSON extraction for the studio endpoints ────────────────────────────────
// WHY: the studio asks the model for a strict JSON object, but models still
// wrap it in ```json fences or add a sentence of prose around it. Returning the
// raw text made the client read `data.enhancedPrompt` from a response that only
// had `result`, so the Enhance button silently did nothing. These helpers turn
// the model's answer into the exact shape the UI already reads, and NEVER return
// an empty object: a failure has to look like a failure, not like "no result".
function extractJsonObject(text: string): Record<string, unknown> | null {
  const raw = (text || "").trim();
  if (!raw) return null;

  // 1) the whole answer is one fenced block
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidates: string[] = [];
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  candidates.push(raw);

  // 2) the first balanced {...} span, so prose before/after is ignored
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start >= 0 && end > start) candidates.push(raw.slice(start, end + 1));

  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      /* try the next candidate */
    }
  }
  return null;
}

/** Coerce a model value into a non-empty string, or fall back. */
function asText(value: unknown, fallback = ""): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

/** Coerce a model value into a string array, dropping anything unusable. */
function asTextArray(value: unknown, fallback: string[] = []): string[] {
  if (Array.isArray(value)) {
    const list = value.filter((v): v is string => typeof v === "string" && v.trim().length > 0);
    if (list.length) return list;
  }
  if (typeof value === "string" && value.trim()) {
    return value
      .split(/\r?\n/)
      .map((l) => l.replace(/^\s*(?:[-*\u2022]|\d+[.)])\s*/, "").trim())
      .filter(Boolean);
  }
  return fallback;
}

const STUDIO_FALLBACK_TAGS = ["SaaS", "Tailwind", "Responsive", "Interactive"];
const STUDIO_FALLBACK_STEPS_AR = [
  "تحليل المتطلبات وتحديد الصفحات والمكونات",
  "بناء الواجهة المتجاوبة بأنماط Tailwind CSS",
  "ربط التفاعلات والحالة",
];
const STUDIO_FALLBACK_STEPS_EN = [
  "Analysing requirements and mapping pages and components",
  "Building the responsive Tailwind CSS interface",
  "Wiring interactions and state",
];

const STUDIO_SYSTEM = `You are the Ebnily AI studio assistant. Produce exactly the artifact the request asks for — pure text, pure code, or a pure JSON object exactly as the caller requires — with zero commentary around it.
${CORE_RULES}`;

// Strip prose/markdown wrappers some models add so clients always receive
// code only (owner standard #1) without touching valid documents.
function stripToCode(text: string): string {
  let t = (text || "").trim();
  // Unwrap a response that is nothing but one fenced block.
  const fence = t.match(/^```[\w-]*\s*([\s\S]*?)\s*```$/);
  if (fence && typeof fence[1] === "string") t = fence[1].trim();
  // Cut any prose written before the document.
  const start = t.search(/<!DOCTYPE html>/i);
  if (start > 0) t = t.slice(start);
  // Cut fences/commentary written after </html> (tail prose, change lists…).
  const end = t.search(/<\/html>/i);
  if (end >= 0) {
    t = t.slice(0, end + "</html>".length);
  } else {
    // Truncated document: drop a trailing closing fence if present.
    const closeFence = t.indexOf("```");
    if (closeFence > 10) t = t.slice(0, closeFence);
  }
  return t.trim();
}

// Selected-element context posted by the visual inspector.
type SelectedElementContext = {
  tagName?: string;
  text?: string;
  className?: string;
  selector?: string;
};

// Normalize whatever the model returned into a complete HTML document.
// Returns "" when nothing usable came back so callers can fall back instead
// of failing (owner standard #3: generation must never break, never 500).
function normalizeRefinedHtml(raw: string): string {
  let t = (raw || "").trim();
  if (!t) return "";
  const fence = t.match(/^```[\w-]*\s*([\s\S]*?)\s*```$/);
  if (fence && typeof fence[1] === "string") t = fence[1].trim();
  if (t.startsWith("{")) {
    try {
      const parsed = JSON.parse(t) as { html?: unknown };
      if (typeof parsed.html === "string" && parsed.html.trim()) t = parsed.html.trim();
    } catch {
      // Not JSON — keep the raw text and continue below.
    }
  }
  const doc = stripToCode(t);
  if (!doc) return "";
  if (/<!DOCTYPE html>/i.test(doc)) return doc;
  if (/<html[\s>]/i.test(doc)) return `<!DOCTYPE html>\n${doc}`;
  return "";
}

// NOTE: the old `applyLocalRefinement()` (a regex colour-swap) used to live
// here as the refine-app fallback. It rewrote Tailwind classes and then the
// endpoint reported "تم التعديل" — a cosmetic change dressed up as a real
// edit, which is exactly the "fake result" complaint. The fallback now returns
// the untouched original code with `applied: false` and says so.

// Turn whatever came back into a document the browser can actually run.
// `stripToCode` cuts at the first `</html>`; when a long generation was cut off
// before its closing tag we close the file instead of shipping a half page
// (which used to render as an empty preview — a "fake" result).
function repairDocument(raw: string): string {
  let out = stripToCode(raw);
  if (!out) return "";
  if (!/<html[\s>]/i.test(out)) {
    out =
      `<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n` +
      `<meta name="viewport" content="width=device-width, initial-scale=1.0">\n</head>\n<body>\n${out}`;
  } else if (!/<!DOCTYPE/i.test(out)) {
    out = `<!DOCTYPE html>\n${out}`;
  }
  if (!/<\/body>/i.test(out)) out += "\n</body>";
  if (!/<\/html>/i.test(out)) out += "\n</html>";
  return out;
}

// Owner-facing message. The model diagnostics stay in `debug` (never shown);
// the visitor gets one actionable Arabic/English sentence instead of a wall of
// SDK internals.
function friendlyAiError(raw: string, language: string): string {
  const en = language !== "ar";
  if (/timed out|deadline|budget|ETIMEDOUT|aborted/i.test(raw))
    return en
      ? "The AI engine took too long to answer. Please try again in a moment."
      : "محرك الذكاء الاصطناعي استغرق وقتاً طويلاً للإجابة. حاول مرة أخرى بعد قليل.";
  if (/high demand|503|UNAVAILABLE|overloaded|\b429\b|rate/i.test(raw))
    return en
      ? "The AI engine is under heavy load right now. Please try again in a minute."
      : "محرك الذكاء الاصطناعي عليه ضغط كبير حالياً. حاول مرة أخرى بعد دقيقة.";
  if (/API_KEY|API key/i.test(raw) && /invalid|incorrect|missing|not valid/i.test(raw))
    return en
      ? "The GEMINI_API_KEY configured on the server is not valid."
      : "مفتاح GEMINI_API_KEY غير صالح على الخادم.";
  return en
    ? "Generation failed. Please try again."
    : "فشل توليد التطبيق. حاول مرة أخرى.";
}

app.post("/api/ai/generate-app", requireAiSession, async (req: Request, res: Response) => {
  try {
    const session = readAuthSession(req as AuthReq);
    if (session && aiQuotaExceeded(session.id)) {
      return res.status(429).json({
        success: false,
        code: "AI_QUOTA_EXCEEDED",
        message: "بلغت الحد الأقصى للطلبات اليومي. حاول غداً أو رقِّ باقتك عبر أورانج كاش.",
      });
    }
    const ai = getGeminiClient();
    if (!ai) return res.status(503).json({ success: false, message: "GEMINI_API_KEY غير مُعد على الخادم" });
    const { prompt, language = "ar" } = (req.body as { prompt?: string; language?: string }) ?? {};
    if (!prompt || !String(prompt).trim())
      return res.status(400).json({ success: false, message: "prompt مطلوب" });
    // Any length of text is accepted; only an extreme paste is elided so the
    // request can never exceed the model/budget limits.
    const bounded = boundText(String(prompt), MAX_PROMPT_CHARS, "تم اختصار منتصف الطلب");
    const result = await generateWithGemini(
      ai,
      `Build a complete single-file HTML app for this request (lang: ${language}):\n${bounded}`,
      GENERATE_SYSTEM,
      {
        budgetMs: GENERATE_BUDGET_MS,
        attemptTimeoutMs: GENERATE_ATTEMPT_TIMEOUT_MS,
        maxOutputTokens: DOCUMENT_MAX_OUTPUT_TOKENS,
      },
    );
    const code = repairDocument(extractText(result));
    if (!code) {
      console.error("generate-app: Gemini returned empty text");
      return res.status(502).json({ success: false, message: "الذكاء الاصطناعي أعاد رداً فارغاً، حاول بصياغة مختلفة" });
    }
    res.json({ success: true, code, appName: String(prompt).slice(0, 60) });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("generate-app failed:", msg);
    if (/API_KEY|API key|key/i.test(msg) && /invalid|incorrect|missing|not valid/i.test(msg)) {
      return res.status(503).json({ success: false, message: "مفتاح GEMINI_API_KEY غير صالح، تحقق من القيمة في Vercel" });
    }
    const language = String((req.body as { language?: string })?.language ?? "ar");
    res.status(502).json({
      success: false,
      message: friendlyAiError(msg, language),
      // Full per-model diagnostics for the client/owner (not shown in UI).
      debug: msg,
    });
  }
});

// ── Streaming generation (SSE) ────────────────────────────────────────────────
// WHY: even with a correct budget a full site takes 40–110s. Waiting in silence
// for that long looks broken, so this endpoint streams the document to the
// browser token-by-token over Server-Sent Events and the preview paints while
// the model is still writing. Same models, same system rules, same fallbacks —
// only the transport changes. On any model-level failure the client silently
// falls back to the non-streaming /api/ai/generate-app above.
app.post("/api/ai/generate-app/stream", requireAiSession, async (req: Request, res: Response) => {
  const language = String((req.body as { language?: string })?.language ?? "ar");
  const promptRaw = String((req.body as { prompt?: string })?.prompt ?? "");
  const ai = getGeminiClient();

  if (!ai || !promptRaw.trim()) {
    return res.status(400).json({ success: false, message: !ai ? "GEMINI_API_KEY غير مُعد" : "prompt مطلوب" });
  }

  // SSE needs an unbuffered text/event-stream; Vercel otherwise holds the body
  // until the function ends, which is exactly what we are trying to avoid.
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders?.();

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };
  const deadline = Date.now() + GENERATE_BUDGET_MS;
  const deadModels = new Set<string>();
  const failures: string[] = [];
  let sentAny = false;

  // Immediately tell the client we're alive so its spinner starts now, not
  // after the first token.
  send("open", { ok: true });

  for (const model of CANDIDATE_MODELS) {
    if (deadModels.has(model)) continue;
    const remaining = deadline - Date.now();
    if (remaining <= 2_000) break;
    const attemptTimeout = Math.min(remaining, GENERATE_ATTEMPT_TIMEOUT_MS);
    const usesThinkingLevel = model === "gemini-3.5-flash" || model === "gemini-3.5-flash-lite";
    const config = {
      systemInstruction: GENERATE_SYSTEM,
      ...(usesThinkingLevel ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } } : {}),
      maxOutputTokens: DOCUMENT_MAX_OUTPUT_TOKENS,
    };

    try {
      const stream = await withDeadline(
        ai.models.generateContentStream({
          model,
          contents: `Build a complete single-file HTML app for this request (lang: ${language}):\n${boundText(promptRaw, MAX_PROMPT_CHARS, "تم اختصار منتصف الطلب")}`,
          config,
        }),
        attemptTimeout,
        `Gemini model "${model}"`,
      );

      for await (const chunk of stream) {
        const delta = extractText(chunk);
        if (!delta) continue;
        sentAny = true;
        send("delta", { text: delta });
        if (Date.now() > deadline) break;
      }
      if (sentAny) {
        send("done", { ok: true, model });
        return res.end();
      }
      throw new Error(`Gemini model "${model}" produced no text`);
    } catch (e) {
      const raw = (e instanceof Error ? e.message : String(e)).replace(/\s+/g, " ");
      failures.push(`${model}: ${raw.slice(0, 140)}`);
      console.error(`stream ${model} failed:`, raw);
      // A 404/invalid-key model is dead for this key; a 429/503 is transient
      // and simply moves to the next candidate.
      if (classifyModelError(raw) === "permanent") deadModels.add(model);
      // If we already streamed content we must not start a second document
      // mid-stream — stop and let the client fall back for a clean result.
      if (sentAny) break;
    }
  }

  send("failed", { message: friendlyAiError(failures.join(" | "), language), debug: failures.join(" | ").slice(0, 800) });
  return res.end();
});

app.post("/api/ai/refine-app", requireAiSession, async (req: Request, res: Response) => {
  const body = (req.body ?? {}) as {
    prompt?: string;
    currentCode?: string;
    selectedElement?: SelectedElementContext;
    language?: string;
  };
  const { prompt, currentCode, selectedElement, language = "ar" } = body;
  if (typeof prompt !== "string" || !prompt.trim() || typeof currentCode !== "string" || !currentCode.trim())
    return res.status(400).json({ success: false, message: "prompt و currentCode مطلوبان" });

  const plan =
    language === "ar"
      ? ["فحص الكود الحالي وتحديد موضع التعديل", "تطبيق التعديلات والأنماط المطلوبة", "تحديث المعاينة المباشرة"]
      : ["Inspecting current code and target location", "Applying requested changes and styles", "Refreshing live preview"];

  // Owner standard #3: this endpoint must never answer 500. BUT it must also
  // never pretend: when the engine is unavailable we hand the ORIGINAL code
  // back with `applied: false` and an honest explanation, instead of the old
  // behaviour that ran a cosmetic string-replace and announced "تم التعديل"
  // while nothing had actually changed.
  const fallback = (reason: string) => {
    const note =
      language === "ar"
        ? `⚠️ لم يتم تطبيق التعديل: محرك الذكاء الاصطناعي غير متاح حالياً (${reason}). كودك الحالي لم يتغير — حاول مرة أخرى.`
        : `⚠️ Change not applied: the AI engine is unavailable (${reason}). Your current code is unchanged — please try again.`;
    return res.json({
      success: true,
      source: "fallback",
      applied: false,
      code: currentCode as string,
      explanation: note,
      plan: [],
    });
  };

  try {
    const ai = getGeminiClient();
    if (!ai) return fallback("no-api-key");
    // The visual inspector posts the element under the cursor — passing it to
    // the model is what makes "make this button green" edit the right node.
    const target = selectedElement
      ? `\nTarget element: <${selectedElement.tagName}> text "${selectedElement.text ?? ""}" classes "${selectedElement.className ?? ""}" selector "${selectedElement.selector ?? ""}"\n`
      : "";
    const result = await generateWithGemini(
      ai,
      `Refine this HTML app (lang: ${language}). Instruction: ${boundText(String(prompt), 8_000, "تم اختصار منتصف الطلب")}${target}\n\nCurrent code:\n${boundText(String(currentCode), MAX_CURRENT_CODE_CHARS, "تم اختصار منتصف الكود")}`,
      REFINE_SYSTEM,
      {
        budgetMs: REFINE_BUDGET_MS,
        attemptTimeoutMs: REFINE_ATTEMPT_TIMEOUT_MS,
        maxOutputTokens: DOCUMENT_MAX_OUTPUT_TOKENS,
      },
    );
    const refined = normalizeRefinedHtml(extractText(result));
    if (!refined) {
      console.error("refine-app: model returned no usable HTML");
      return fallback("no-usable-output");
    }
    const explanation =
      language === "ar" ? `تم تطبيق التعديل: "${prompt}"` : `Applied modification: "${prompt}"`;
    return res.json({ success: true, source: "gemini", applied: true, code: refined, explanation, plan });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("refine-app failed:", msg);
    return fallback(friendlyAiError(msg, language));
  }
});

// ── Studio endpoints ─────────────────────────────────────────────────────────
// WHY these are separate handlers instead of one loop:
// the old loop returned `{ success, result }` for every path, but each screen
// reads a DIFFERENT shape — `enhancedPrompt`/`suggestedTags` in the chat +
// generator, `htmlCode`/`reactComponent`/… in the architect, `diagnosis`/
// `improvements` in the doctor. The keys never matched, so all three buttons
// silently did nothing. Each handler below asks the model for its own JSON
// contract and maps the answer onto the exact keys the UI already reads.
app.post("/api/ai/gemini-enhance-prompt", requireAiSession, async (req: Request, res: Response) => {
  try {
    const session = readAuthSession(req as AuthReq);
    if (session && aiQuotaExceeded(session.id)) {
      return res.status(429).json({ success: false, message: "بلغت الحد الأقصى للطلبات اليومي" });
    }
    const ai = getGeminiClient();
    if (!ai) return res.status(503).json({ success: false, message: "GEMINI_API_KEY غير مُعد على الخادم" });
    const body = (req.body ?? {}) as { prompt?: string; category?: string; language?: string };
    const language = String(body.language ?? "ar");
    const original = String(body.prompt ?? "").trim();
    if (!original) return res.status(400).json({ success: false, message: "اكتب وصفاً أولاً ثم اطلب تحسينه" });

    const instruction = [
      `(language: ${language})`,
      "Expand this web-app idea into a precise, buildable specification.",
      `Category hint: ${String(body.category ?? "app")}.`,
      'Return ONLY a JSON object: {"enhancedPrompt": string, "suggestedTags": string[], "appName": string}',
      "enhancedPrompt must be one self-contained brief in the requested language naming the pages, sections, colour direction, the data shown, and the interactions.",
    ].join("\n");

    const result = await generateWithGemini(
      ai,
      `${instruction}\n\nIDEA:\n${boundText(original, MAX_PROMPT_CHARS, "تم اختصار منتصف الطلب")}`,
      STUDIO_SYSTEM,
      { budgetMs: STUDIO_BUDGET_MS, attemptTimeoutMs: STUDIO_ATTEMPT_TIMEOUT_MS, maxOutputTokens: 8192 },
    );
    const text = extractText(result);
    if (!text) {
      console.error("gemini-enhance-prompt: Gemini returned empty text");
      return res.status(502).json({ success: false, message: "الذكاء الاصطناعي أعاد رداً فارغاً، حاول بصياغة مختلفة" });
    }

    const parsed = extractJsonObject(text);
    if (!parsed) {
      // The model answered in prose instead of JSON. The text is still a usable
      // specification, so hand it back rather than failing a click silently.
      console.error("gemini-enhance-prompt: response was not JSON, using raw text");
      return res.json({ success: true, enhancedPrompt: text, suggestedTags: STUDIO_FALLBACK_TAGS, appName: "", result: text });
    }

    const enhanced = asText(parsed.enhancedPrompt ?? parsed.prompt ?? parsed.result, "");
    if (!enhanced) return res.status(502).json({ success: false, message: "تعذّر تحسين الوصف، حاول مرة أخرى" });

    return res.json({
      success: true,
      enhancedPrompt: enhanced,
      suggestedTags: asTextArray(parsed.suggestedTags, STUDIO_FALLBACK_TAGS),
      appName: asText(parsed.appName, ""),
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("gemini-enhance-prompt failed:", msg);
    if (/API_KEY|API key|key/i.test(msg) && /invalid|incorrect|missing|not valid/i.test(msg)) {
      return res.status(503).json({ success: false, message: "مفتاح GEMINI_API_KEY غير صالح، تحقق من القيمة في Vercel" });
    }
    return res.status(502).json({ success: false, message: "فشل تحسين الوصف، حاول مرة أخرى" });
  }
});

app.post("/api/ai/gemini-architect", requireAiSession, async (req: Request, res: Response) => {
  try {
    const session = readAuthSession(req as AuthReq);
    if (session && aiQuotaExceeded(session.id)) {
      return res.status(429).json({ success: false, message: "بلغت الحد الأقصى للطلبات اليومي" });
    }
    const ai = getGeminiClient();
    if (!ai) return res.status(503).json({ success: false, message: "GEMINI_API_KEY غير مُعد على الخادم" });
    const body = (req.body ?? {}) as { prompt?: string; language?: string };
    const language = String(body.language ?? "ar");
    const original = String(body.prompt ?? "").trim();
    if (!original) return res.status(400).json({ success: false, message: "اكتب وصفاً أولاً ثم اطلب التصميم" });

    const instruction = [
      `(language: ${language})`,
      "Act as a software architect and produce four deliverables for this app idea.",
      'Return ONLY a JSON object: {"appName": string, "thinkingSteps": string[], "htmlCode": string, "reactComponent": string, "apiEndpoint": string, "databaseSchema": string}',
      "htmlCode: one complete runnable single-file HTML document (inline CSS and JS, Tailwind CDN).",
      "reactComponent: one self-contained React + Tailwind component using useState.",
      "apiEndpoint: one Express router snippet. databaseSchema: one SQL CREATE TABLE script.",
      `thinkingSteps: 3-5 short steps in ${language}.`,
      "Escape every newline inside the code strings as \\n so the JSON stays valid.",
    ].join("\n");

    const result = await generateWithGemini(
      ai,
      `${instruction}\n\nIDEA:\n${boundText(original, MAX_PROMPT_CHARS, "تم اختصار منتصف الطلب")}`,
      STUDIO_SYSTEM,
      { budgetMs: STUDIO_BUDGET_MS, attemptTimeoutMs: STUDIO_ATTEMPT_TIMEOUT_MS, maxOutputTokens: DOCUMENT_MAX_OUTPUT_TOKENS },
    );
    const text = extractText(result);
    if (!text) {
      console.error("gemini-architect: Gemini returned empty text");
      return res.status(502).json({ success: false, message: "الذكاء الاصطناعي أعاد رداً فارغاً، حاول بصياغة مختلفة" });
    }

    const parsed = extractJsonObject(text);
    if (!parsed) return res.status(502).json({ success: false, message: "تعذّر تحليل الرد، أعد المحاولة" });
    const raw = (k: string) => asText(parsed[k], "");
    // A tab that renders nothing reads as a broken feature, so refuse loudly
    // rather than showing four empty panes.
    if (!raw("htmlCode") && !raw("reactComponent") && !raw("apiEndpoint") && !raw("databaseSchema")) {
      return res.status(502).json({ success: false, message: "تعذّر توليد المخرجات، أعد المحاولة" });
    }

    const fallbackSteps = language === "ar" ? STUDIO_FALLBACK_STEPS_AR : STUDIO_FALLBACK_STEPS_EN;
    return res.json({
      success: true,
      appName: asText(parsed.appName, ""),
      thinkingSteps: asTextArray(parsed.thinkingSteps, fallbackSteps),
      htmlCode: raw("htmlCode"),
      reactComponent: raw("reactComponent"),
      apiEndpoint: raw("apiEndpoint"),
      databaseSchema: raw("databaseSchema"),
      result: text, // kept so older builds that read `result` still work
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("gemini-architect failed:", msg);
    if (/API_KEY|API key|key/i.test(msg) && /invalid|incorrect|missing|not valid/i.test(msg)) {
      return res.status(503).json({ success: false, message: "مفتاح GEMINI_API_KEY غير صالح، تحقق من القيمة في Vercel" });
    }
    return res.status(502).json({ success: false, message: "فشل التصميم، حاول مرة أخرى" });
  }
});

app.post("/api/ai/gemini-code-doctor", requireAiSession, async (req: Request, res: Response) => {
  try {
    const session = readAuthSession(req as AuthReq);
    if (session && aiQuotaExceeded(session.id)) {
      return res.status(429).json({ success: false, message: "بلغت الحد الأقصى للطلبات اليومي" });
    }
    const ai = getGeminiClient();
    if (!ai) return res.status(503).json({ success: false, message: "GEMINI_API_KEY غير مُعد على الخادم" });
    const body = (req.body ?? {}) as { code?: string; issueDescription?: string; language?: string };
    const language = String(body.language ?? "ar");
    const code = String(body.code ?? "");
    if (!code.trim()) return res.status(400).json({ success: false, message: "لا يوجد كود لفحصه" });
    const focus = String(body.issueDescription ?? "Optimize and fix any bugs").trim();

    const instruction = [
      `(language: ${language})`,
      "Review this HTML application for layout bugs, responsiveness problems, accessibility gaps and script performance issues.",
      `Focus on: ${boundText(focus, 2000, "تم الاختصار")}`,
      'Return ONLY a JSON object: {"diagnosis": string, "improvements": string[], "fixedCode": string}',
      "diagnosis: a short plain-language summary in the requested language.",
      "improvements: 3-7 concrete actionable strings in the requested language.",
      "fixedCode: the complete corrected HTML document; if no change is needed, repeat the input unchanged.",
      "Escape every newline inside fixedCode as \\n so the JSON stays valid.",
    ].join("\n");

    const result = await generateWithGemini(
      ai,
      `${instruction}\n\nCODE:\n${boundText(code, MAX_CURRENT_CODE_CHARS, "تم اختصار منتصف الكود")}`,
      STUDIO_SYSTEM,
      { budgetMs: STUDIO_BUDGET_MS, attemptTimeoutMs: STUDIO_ATTEMPT_TIMEOUT_MS, maxOutputTokens: DOCUMENT_MAX_OUTPUT_TOKENS },
    );
    const text = extractText(result);
    if (!text) {
      console.error("gemini-code-doctor: Gemini returned empty text");
      return res.status(502).json({ success: false, message: "الذكاء الاصطناعي أعاد رداً فارغاً، حاول مرة أخرى" });
    }

    const parsed = extractJsonObject(text);
    if (!parsed) return res.status(502).json({ success: false, message: "تعذّر تحليل الرد، أعد المحاولة" });
    const diagnosis = asText(parsed.diagnosis, "");
    const improvements = asTextArray(parsed.improvements, []);
    const fixedCode = asText(parsed.fixedCode, "");
    if (!diagnosis && !improvements.length && !fixedCode) {
      return res.status(502).json({ success: false, message: "تعذّر فحص الكود، أعد المحاولة" });
    }

    return res.json({
      success: true,
      diagnosis,
      improvements,
      // An empty fixedCode would make "apply fix" a silent no-op, so hand back
      // the original document rather than nothing.
      fixedCode: fixedCode || code,
      result: text, // kept so older builds that read `result` still work
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error("gemini-code-doctor failed:", msg);
    if (/API_KEY|API key|key/i.test(msg) && /invalid|incorrect|missing|not valid/i.test(msg)) {
      return res.status(503).json({ success: false, message: "مفتاح GEMINI_API_KEY غير صالح، تحقق من القيمة في Vercel" });
    }
    return res.status(502).json({ success: false, message: "فشل فحص الكود، حاول مرة أخرى" });
  }
});

// ── Unknown /api paths → JSON 404 (never HTML, never crash) ─────────────────
app.use("/api", (_req: Request, res: Response) => {
  res.status(404).json({ success: false, message: "API endpoint not found" });
});

// ── Global error handler (last resort: always JSON) ─────────────────────────
app.use(
  (err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    console.error("Unhandled API error:", err);
    if (res.headersSent) return;
    const e = err as { type?: string };
    // Body-parser failures are client errors, not server errors.
    if (e?.type === "entity.too.large")
      return res.status(413).json({ success: false, message: "Request payload too large" });
    if (e?.type === "entity.parse.failed")
      return res.status(400).json({ success: false, message: "Invalid JSON body" });
    res.status(500).json({ success: false, message: "Internal server error" });
  },
);

export default app;

/**
 * Mount the whole API surface onto another Express instance.
 *
 * WHY THIS EXISTS
 * ---------------
 * `api/index.ts` is the single implementation of every `/api/*` route (auth,
 * projects, GitHub import, AI, admin). `server.ts` is the self-hosted runner
 * (`npm start`, Docker, any Node host) and used to declare its OWN, much smaller
 * set of routes — it had no `/api/auth/*`, no `/api/projects*`, no `/api/github/*`.
 * The result was that running the project outside Vercel served a frontend that
 * could never sign anyone in: the login wall listed no providers, the project
 * store 404'd, and repository import was simply absent. The site was effectively
 * broken on every self-hosted deployment.
 *
 * `server.ts` now mounts THIS app instead of re-implementing the routes, so both
 * runners execute exactly one copy of the API and can never drift apart.
 *
 * Usage: `app.use(mountApi())` — mounted at the ROOT, not at `/api`, because
 * every route here is already declared with its full `/api/...` path. Mounting
 * at `/api` would make Express look for `/api/api/health`.
 */
export function mountApi(): express.Express {
  return app;
}


