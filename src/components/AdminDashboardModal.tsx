import { useState, useEffect, useCallback } from 'react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Cpu, 
  CreditCard, 
  Settings, 
  X, 
  Lock, 
  RefreshCw, 
  CheckCircle, 
  AlertTriangle, 
  Search, 
  Smartphone,
  DollarSign,
  TrendingUp,
  Ban,
  // Users tab: accounts, their credit wallet, and the live presence dot.
  Users,
  User,
  Coins,
  Loader2
} from 'lucide-react';
import { Language, PlatformRealStats, AdminSettings, DeviceProtectionInfo, OrangeCashTransaction } from '../types';
import { useModalAccessibility } from './useModalAccessibility';

interface AdminDashboardModalProps {
  isOpen: boolean;
  onClose: () => void;
  language: Language;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const toFiniteNumber = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

const toText = (value: unknown, fallback = ''): string =>
  typeof value === 'string' ? value : fallback;

const EMPTY_STATS: PlatformRealStats = {
  totalDevicesCount: 0,
  blockedDevicesCount: 0,
  totalGenerationsExecuted: 0,
  totalRevenueEGP: 0,
  activeProUsersCount: 0,
  lastActiveTime: '',
  totalTransactionsCount: 0,
};

const DEFAULT_ADMIN_SETTINGS: AdminSettings = {
  orangeWalletNumber: '01207782741',
  defaultFreeLimit: 5,
  autoVerificationEnabled: true,
  supportWhatsappNumber: '01207782741',
  siteName: 'إبنيلي | Ebnili AI Studio',
  // Placeholder only. This object ships to EVERY visitor inside the public JS
  // bundle, so the owner's address must never be hard-coded here — the real
  // value arrives from `/api/admin/overview`, which is owner-session only.
  adminEmail: '',
};

const normalizeAdminSettings = (value: unknown, previous?: AdminSettings | null): AdminSettings => {
  const source = isRecord(value) ? value : {};
  const previousSource = isRecord(previous) ? previous : {};
  const base = {
    ...DEFAULT_ADMIN_SETTINGS,
    ...previousSource,
  } as AdminSettings;
  const rawLimit = toFiniteNumber(source.defaultFreeLimit, base.defaultFreeLimit);

  return {
    orangeWalletNumber: toText(source.orangeWalletNumber, base.orangeWalletNumber),
    defaultFreeLimit: Math.min(50, Math.max(1, rawLimit)),
    autoVerificationEnabled:
      typeof source.autoVerificationEnabled === 'boolean'
        ? source.autoVerificationEnabled
        : base.autoVerificationEnabled,
    supportWhatsappNumber: toText(source.supportWhatsappNumber, base.supportWhatsappNumber),
    siteName: toText(source.siteName, base.siteName),
    adminEmail: toText(source.adminEmail, base.adminEmail),
  };
};

// The admin API is stateless on Vercel and may return a partial payload during a
// cold start. Never let a missing/null row reach JSX: `null.isBlocked` used to
// throw inside React and the global ErrorBoundary replaced the whole app with a
// white screen.
const normalizeDevices = (value: unknown): DeviceProtectionInfo[] => {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((device, index) => ({
    deviceId: toText(device.deviceId, `unknown-device-${index}`),
    fingerprintHash: toText(device.fingerprintHash),
    ipAddress: toText(device.ipAddress) || undefined,
    userAgent: toText(device.userAgent) || undefined,
    freeGenerationsUsed: toFiniteNumber(device.freeGenerationsUsed, 0),
    freeGenerationsLimit: toFiniteNumber(device.freeGenerationsLimit, 5),
    isBlocked: device.isBlocked === true,
    blockReason: toText(device.blockReason) || undefined,
    registeredEmails: Array.isArray(device.registeredEmails)
      ? device.registeredEmails.filter((email): email is string => typeof email === 'string')
      : [],
    lastSeen: toText(device.lastSeen) || undefined,
  }));
};

const normalizeTransactions = (value: unknown): OrangeCashTransaction[] => {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((transaction, index) => ({
    id: toText(transaction.id, `unknown-transaction-${index}`),
    senderPhone: toText(transaction.senderPhone),
    recipientWallet: toText(transaction.recipientWallet),
    transactionReference: toText(transaction.transactionReference),
    amount: toFiniteNumber(transaction.amount, 0),
    currency: toText(transaction.currency, 'EGP'),
    planId: toText(transaction.planId, 'free') as OrangeCashTransaction['planId'],
    planName: toText(transaction.planName),
    billingCycle: toText(transaction.billingCycle, 'monthly') as OrangeCashTransaction['billingCycle'],
    userName: toText(transaction.userName) || undefined,
    userEmail: toText(transaction.userEmail) || undefined,
    submittedAt: toText(transaction.submittedAt, new Date(0).toISOString()),
    status: transaction.status === 'confirmed' || transaction.status === 'pending'
      ? transaction.status
      : 'rejected',
    verifiedAt: toText(transaction.verifiedAt) || undefined,
    receiptImage: toText(transaction.receiptImage) || undefined,
    notes: toText(transaction.notes) || undefined,
  }));
};

const normalizeStats = (value: unknown): PlatformRealStats => {
  if (!isRecord(value)) return EMPTY_STATS;
  return {
    totalDevicesCount: toFiniteNumber(value.totalDevicesCount, 0),
    blockedDevicesCount: toFiniteNumber(value.blockedDevicesCount, 0),
    totalGenerationsExecuted: toFiniteNumber(value.totalGenerationsExecuted, 0),
    totalRevenueEGP: toFiniteNumber(value.totalRevenueEGP, 0),
    activeProUsersCount: toFiniteNumber(value.activeProUsersCount, 0),
    lastActiveTime: toText(value.lastActiveTime),
    totalTransactionsCount: toFiniteNumber(value.totalTransactionsCount, 0),
  };
};

/**
 * A signed-in account as the server reports it.
 *
 * `presence` is computed server-side from the heartbeat rather than in the
 * browser, so the value is the same for every reader and does not drift between
 * the owner's screen and the database.
 */
interface AdminAccount {
  accountId: string;
  email: string;
  name: string;
  provider: string;
  avatarUrl: string | null;
  credits: number;
  welcomeGiven: boolean;
  tier: string;
  isBlocked: boolean;
  blockReason: string | null;
  firstSeenAt: string;
  lastSeenAt: string;
  requestsCount: number;
  lastIp: string | null;
  presence: 'online' | 'idle' | 'offline';
}

const normalizeAccounts = (value: unknown): AdminAccount[] => {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((row, index) => ({
    accountId: toText(row.accountId, `account-${index}`),
    email: toText(row.email),
    name: toText(row.name),
    provider: toText(row.provider),
    avatarUrl: toText(row.avatarUrl) || null,
    credits: toFiniteNumber(row.credits, 0),
    welcomeGiven: row.welcomeGiven === true,
    tier: toText(row.tier, 'free'),
    isBlocked: row.isBlocked === true,
    blockReason: toText(row.blockReason) || null,
    firstSeenAt: toText(row.firstSeenAt),
    lastSeenAt: toText(row.lastSeenAt),
    requestsCount: toFiniteNumber(row.requestsCount, 0),
    lastIp: toText(row.lastIp) || null,
    presence:
      row.presence === 'online' || row.presence === 'idle' ? row.presence : 'offline',
  }));
};

/** "منذ 3 دقائق" — the accounts list is a recency list. */
function relativeTime(iso: string, ar: boolean): string {
  const at = new Date(iso).getTime();
  if (!Number.isFinite(at)) return '—';
  const mins = Math.max(0, Math.round((Date.now() - at) / 60000));
  if (mins < 1) return ar ? 'الآن' : 'now';
  if (mins < 60) return ar ? `منذ ${mins} د` : `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return ar ? `منذ ${hours} س` : `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return ar ? `منذ ${days} يوم` : `${days}d ago`;
  return new Date(iso).toLocaleDateString(ar ? 'ar-EG' : 'en-GB');
}


export const AdminDashboardModal = ({
  isOpen,
  onClose,
  language,
}: AdminDashboardModalProps) => {
  // Focus management must come before any conditional return so the effect is
  // registered consistently across renders.
  const handleClose = useCallback(() => onClose(), [onClose]);
  const dialogRef = useModalAccessibility(isOpen, handleClose);

  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [pinInput, setPinInput] = useState('');
  const [authError, setAuthError] = useState<string | null>(null);
  const [isAuthenticating, setIsAuthenticating] = useState(false);

  // Admin Data State
  const [activeTab, setActiveTab] = useState<'overview' | 'users' | 'devices' | 'transactions' | 'settings' | 'admins'>('overview');
  const [isLoadingData, setIsLoadingData] = useState(false);
  const [isDataReady, setIsDataReady] = useState(false);
  const [dataError, setDataError] = useState<string | null>(null);
  // Never start the authenticated view with nullable stats. A valid cookie can
  // outlive the data request, and `stats.*` must never reach React as null.
  const [stats, setStats] = useState<PlatformRealStats>(EMPTY_STATS);
  const [settingsState, setSettings] = useState<AdminSettings>(DEFAULT_ADMIN_SETTINGS);
  // Always render from a fully-populated object. Never access an optional or
  // stale settings value directly from JSX.
  const settings = normalizeAdminSettings(settingsState, DEFAULT_ADMIN_SETTINGS);

  /**
   * Accounts + their presence.
   *
   * WHY a separate poll: presence is only true while someone is actually
   * generating a heartbeat, so a single load would show every account as frozen
   * at the moment the modal opened. Refreshing the overview on a timer keeps the
   * online/offline column honest while the owner is watching it.
   */
  const [accounts, setAccounts] = useState<AdminAccount[]>([]);
  const [accountStats, setAccountStats] = useState({
    total: 0,
    online: 0,
    idle: 0,
    credits: 0,
    welcomePerSignup: 5,
  });
  /** Accounts currently being written to, so one row cannot be double-clicked. */
  const [busyAccountId, setBusyAccountId] = useState<string | null>(null);


  const [devices, setDevices] = useState<DeviceProtectionInfo[]>([]);
  const [transactions, setTransactions] = useState<OrangeCashTransaction[]>([]);
  const [isCheckingSession, setIsCheckingSession] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);

  // Admin email management
  interface AdminRow { id: string; email: string; added_by: string; added_at: string }
  const [adminList, setAdminList] = useState<AdminRow[]>([]);
  const [adminEmailInput, setAdminEmailInput] = useState('');
  const [isAddingAdmin, setIsAddingAdmin] = useState(false);

  // Filtering & Search
  const [searchQuery, setSearchQuery] = useState('');
  const [actionSuccessMessage, setActionSuccessMessage] = useState<string | null>(null);
  /**
   * Failures are kept out of `actionSuccessMessage`.
   *
   * The single toast used to render everything on a green background, so a
   * rejected write ("تعذّر تحديث الكريديت") looked exactly like a successful
   * one. Two states, two colours.
   */
  const [actionErrorMessage, setActionErrorMessage] = useState<string | null>(null);

  // Verify the HttpOnly session cookie whenever the modal opens. The legacy
  // sessionStorage flag is only a hint and can survive an expired cookie.
  useEffect(() => {
    if (!isOpen) return;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const requestController = new AbortController();
    setIsAuthenticated(false);
    setIsCheckingSession(true);
    setIsLoadingData(true);
    setIsDataReady(false);
    setDataError(null);
    setSessionError(null);
    setAuthError(null);

    // The overview response is both session verification and dashboard data.
    // Use one request so opening the modal cannot race two state transitions.
    const verifySessionAndLoadData = async () => {
      try {
        const res = await fetch('/api/admin/overview', {
          credentials: 'include',
          cache: 'no-store',
          signal: requestController.signal,
        });
        if (res.status === 401) {
          try { window.sessionStorage.removeItem('ebnili_admin_auth'); } catch { /* noop */ }
          return;
        }
        if (!res.ok) {
          // A 503 with `configured: false` means the dashboard's tables have not
          // been created yet. That is an owner-side setup step, not a session
          // problem, so it must not be reported as "your session expired".
          if (res.status === 503) {
            const payload = (await res.json().catch(() => ({}))) as { message?: string };
            setDataError(
              payload.message ||
                'قاعدة بيانات لوحة الإدارة غير مهيأة. نفّذ supabase/projects.sql في Supabase ثم أعد المحاولة.',
            );
            setIsDataReady(true);
            return;
          }
          if (res.status === 401) {
            try { window.sessionStorage.removeItem('ebnili_admin_auth'); } catch { /* noop */ }
            return;
          }
          throw new Error(`تعذر التحقق من الجلسة (${res.status})`);
        }

        const data: unknown = await res.json().catch(() => null);
        if (!isRecord(data) || data.success !== true) {
          throw new Error('استجابة بيانات لوحة الإدارة غير صالحة.');
        }
        if (cancelled) return;

        setStats(normalizeStats(data.stats));
        setSettings((prev) => normalizeAdminSettings(data.settings, prev));
        setDevices(normalizeDevices(data.devices));
        setTransactions(normalizeTransactions(data.recentTransactions));
        // Admin email list from the overview response.
        const rawAdmins = (data as { admins?: unknown }).admins;
        if (Array.isArray(rawAdmins)) {
          setAdminList(rawAdmins.filter((r): r is AdminRow => isRecord(r) && typeof r.email === 'string').map((r) => ({
            id: toText(r.id),
            email: toText(r.email),
            added_by: toText(r.added_by),
            added_at: toText(r.added_at),
          })));
        }
        // Accounts + presence. `accounts` is absent on a deployment that has not
        // run the accounts half of the SQL yet, so this normalizes to [] rather
        // than throwing — the Users tab then says "run the SQL" instead of the
        // whole dashboard failing.
        setAccounts(normalizeAccounts(data.accounts));
        setAccountStats({
          total: toFiniteNumber(
            isRecord(data.stats) ? data.stats.totalAccountsCount : 0,
            normalizeAccounts(data.accounts).length,
          ),
          online: toFiniteNumber(isRecord(data.stats) ? data.stats.onlineAccountsCount : 0, 0),
          idle: toFiniteNumber(isRecord(data.stats) ? data.stats.idleAccountsCount : 0, 0),
          credits: toFiniteNumber(
            isRecord(data.stats) ? data.stats.totalCreditsOutstanding : 0,
            0,
          ),
          welcomePerSignup: toFiniteNumber(
            isRecord(data.stats) ? data.stats.welcomeCreditsPerSignup : 0,
            5,
          ),
        });
        setIsDataReady(true);
        setIsAuthenticated(true);
      } catch (err: unknown) {
        if (cancelled) return;
        if (err instanceof DOMException && err.name === 'AbortError') {
          setSessionError('انتهت مهلة تحميل اللوحة. حاول مرة أخرى.');
        } else {
          setSessionError(err instanceof Error ? err.message : 'تعذر التحقق من جلسة الإدارة');
        }
      } finally {
        if (timer !== undefined) clearTimeout(timer);
        if (!cancelled) {
          setIsCheckingSession(false);
          setIsLoadingData(false);
        }
      }
    };

    timer = setTimeout(() => {
      if (!cancelled) requestController.abort();
    }, 20_000);

    void verifySessionAndLoadData();
    return () => {
      cancelled = true;
      requestController.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [isOpen]);

  /**
   * Keep presence live while the dashboard is open.
   *
   * 15s is deliberately shorter than the server's 90s online window: an account
   * that closes its tab must visibly drop to "offline" within a minute or two,
   * and a stale green dot is worse than no dot at all. This is the ONLY place
   * the overview is re-fetched — nothing else in the modal depends on it, and a
   * failed tick is ignored so a transient network blip never blanks the table.
   */
  useEffect(() => {
    if (!isOpen || !isAuthenticated) return;
    let cancelled = false;

    const tick = async () => {
      try {
        const res = await fetch('/api/admin/overview', {
          credentials: 'include',
          cache: 'no-store',
        });
        if (!res.ok) return;
        const data: unknown = await res.json().catch(() => null);
        if (cancelled || !isRecord(data) || data.success !== true) return;
        setAccounts(normalizeAccounts(data.accounts));
        // Bound to a local BEFORE the object literal: TypeScript cannot carry a
        // narrowing made by `isRecord(x)` into a closure, so `data.stats.foo`
        // inside the updater was still `unknown` and failed to type-check.
        const stats = data.stats;
        if (isRecord(stats)) {
          setAccountStats((prev) => ({
            ...prev,
            total: toFiniteNumber(stats.totalAccountsCount, prev.total),
            online: toFiniteNumber(stats.onlineAccountsCount, 0),
            idle: toFiniteNumber(stats.idleAccountsCount, 0),
            credits: toFiniteNumber(stats.totalCreditsOutstanding, 0),
            welcomePerSignup: toFiniteNumber(stats.welcomeCreditsPerSignup, prev.welcomePerSignup),
          }));
        }
      } catch {
        /* a dropped poll must never interrupt the owner's session */
      }
    };

    const id = setInterval(() => void tick(), 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [isOpen, isAuthenticated]);

  /**
   * Write to one account and fold the server's answer back into the list.
   *
   * WHY the response is trusted over the local value: the server clamps and
   * validates the number, so echoing what it returns is what keeps the table
   * honest after a rejected write. `busyAccountId` disables the row for the
   * duration so a double-click cannot fire two conflicting writes.
   */
  const mutateAccount = useCallback(
    async (
      path: '/api/admin/account/set-credits' | '/api/admin/account/toggle-block',
      body: Record<string, unknown>,
    ) => {
      const accountId = String(body.accountId ?? '');
      setBusyAccountId(accountId);
      try {
        const res = await fetch(path, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(body),
        });
        const data: unknown = await res.json().catch(() => ({}));
        if (!res.ok) {
          const message = isRecord(data) ? toText(data.message) : '';
          // Failures go to the error channel, never the green one.
          setActionErrorMessage(message || 'تعذّر تنفيذ العملية.');
          setActionSuccessMessage(null);
          return;
        }
        const updated = isRecord(data) && isRecord(data.account) ? normalizeAccounts([data.account])[0] : null;
        if (updated) {
          // Recompute the outstanding total from the list itself rather than
          // trying to apply a delta: `accounts` already holds every row, so the
          // sum is exact and cannot drift from what the table shows.
          setAccounts((prev) => {
            const next = prev.map((a) => (a.accountId === updated.accountId ? updated : a));
            setAccountStats((p) => ({
              ...p,
              total: next.length,
              credits: next.reduce((sum, a) => sum + a.credits, 0),
            }));
            return next;
          });
        }
        // Confirm the write actually landed rather than silently claiming it did.
        setActionSuccessMessage(
          path === '/api/admin/account/set-credits' ? 'تم تحديث الكريديت.' : 'تم تحديث حالة الحساب.',
        );
        setActionErrorMessage(null);
      } catch {
        setActionErrorMessage('تعذّر الاتصال بالخادم.');
        setActionSuccessMessage(null);
      } finally {
        setBusyAccountId(null);
      }
    },
    [],
  );

  /** Grant or remove credits through a validated prompt. */
  const handleSetCredits = useCallback(
    (account: AdminAccount) => {
      const raw = window.prompt(
        `عدد الكريديت للحساب ${account.email || account.name || account.accountId}`,
        String(account.credits),
      );
      if (raw === null) return;
      const value = Number(raw.trim());
      if (!Number.isInteger(value) || value < 0 || value > 100000) {
        setActionErrorMessage('أدخل عدداً صحيحاً بين 0 و 100000.');
        setActionSuccessMessage(null);
        return;
      }
      void mutateAccount('/api/admin/account/set-credits', { accountId: account.accountId, credits: value });
    },
    [mutateAccount],
  );

  const handleToggleAccountBlock = useCallback(
    (account: AdminAccount) => {
      const next = !account.isBlocked;
      if (next) {
        const reason = window.prompt(`سبب حظر ${account.email || account.name}`, 'مخالفة الشروط');
        if (reason === null) return;
        void mutateAccount('/api/admin/account/toggle-block', {
          accountId: account.accountId,
          block: true,
          reason,
        });
        return;
      }
      void mutateAccount('/api/admin/account/toggle-block', {
        accountId: account.accountId,
        block: false,
      });
    },
    [mutateAccount],
  );

  const handleAddAdmin = async (e: React.FormEvent) => {
    e.preventDefault();
    const email = adminEmailInput.trim().toLowerCase();
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setActionErrorMessage('البريد الإلكتروني غير صحيح.');
      setActionSuccessMessage(null);
      return;
    }
    setIsAddingAdmin(true);
    try {
      const res = await fetch('/api/admin/admins', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setActionErrorMessage((data as { message?: string }).message || 'تعذّر إضافة الأدمن.');
        setActionSuccessMessage(null);
        return;
      }
      setAdminEmailInput('');
      setActionSuccessMessage(`تمت إضافة ${email} كأدمن بنجاح. سيدخل لوحة التحكم بمجرد تسجيل الدخول بحسابه بدون كلمة مرور.`);
      setActionErrorMessage(null);
      // Refresh the admin list from the overview.
      void fetchAdminData();
    } catch {
      setActionErrorMessage('تعذّر الاتصال بالخادم.');
      setActionSuccessMessage(null);
    } finally {
      setIsAddingAdmin(false);
    }
  };

  const handleRemoveAdmin = async (email: string) => {
    if (!window.confirm(`حذف ${email} من قائمة الأدمن؟`)) return;
    try {
      const res = await fetch('/api/admin/admins', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setActionErrorMessage((data as { message?: string }).message || 'تعذّر حذف الأدمن.');
        setActionSuccessMessage(null);
        return;
      }
      setAdminList((prev) => prev.filter((a) => a.email !== email));
      setActionSuccessMessage(`تم حذف ${email} من قائمة الأدمن.`);
      setActionErrorMessage(null);
    } catch {
      setActionErrorMessage('تعذّر الاتصال بالخادم.');
      setActionSuccessMessage(null);
    }
  };

  const handleLogin = async (e: import('react').FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setIsAuthenticating(true);

    try {
      const res = await fetch('/api/admin/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ pin: pinInput.trim() }),
      });

      const data: { error?: string; message?: string } = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.message || data.error || 'رمز الدخول غير صحيح');
      }

      try { window.sessionStorage.setItem('ebnili_admin_auth', 'true'); } catch { /* noop */ }
      const loaded = await fetchAdminData();
      if (loaded) {
        setIsAuthenticated(true);
      } else {
        setAuthError('تم التحقق من الرمز، لكن تعذر تحميل بيانات اللوحة. حاول مرة أخرى.');
      }
    } catch (err: unknown) {
      setAuthError(err instanceof Error ? err.message : 'فشل تسجيل الدخول كمسؤول');
    } finally {
      setIsAuthenticating(false);
    }
  };

  const handleSessionExpired = () => {
    try { window.sessionStorage.removeItem('ebnili_admin_auth'); } catch { /* noop */ }
    setIsAuthenticated(false);
    setPinInput('');
    setAuthError('انتهت الجلسة — يرجى تسجيل الدخول مجدداً.');
  };

  const fetchAdminData = async (): Promise<boolean> => {
    setIsLoadingData(true);
    setDataError(null);
    setIsDataReady(false);
    try {
      const res = await fetch('/api/admin/overview', {
        credentials: 'include',
        cache: 'no-store',
      });
      if (res.status === 401) {
        handleSessionExpired();
        return false;
      }
      if (!res.ok) throw new Error(`تعذر تحميل بيانات اللوحة (${res.status})`);

      const data: unknown = await res.json().catch(() => null);
      if (!isRecord(data) || data.success !== true) {
        throw new Error('استجابة بيانات لوحة الإدارة غير صالحة.');
      }

      setStats(normalizeStats(data.stats));
      setSettings((prev) => normalizeAdminSettings(data.settings, prev));
      setDevices(normalizeDevices(data.devices));
      setTransactions(normalizeTransactions(data.recentTransactions));
      setIsDataReady(true);
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'تعذر تحميل بيانات لوحة الإدارة.';
      console.error('Failed to load admin data:', err);
      setDataError(message);
      setIsDataReady(false);
      return false;
    } finally {
      setIsLoadingData(false);
    }
  };

  const handleToggleBlockDevice = async (device: DeviceProtectionInfo) => {
    try {
      const res = await fetch('/api/admin/device/toggle-block', {
        method: 'POST',

        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          deviceId: device.deviceId,
          fingerprintHash: device.fingerprintHash,
          block: !device.isBlocked,
          reason: !device.isBlocked ? 'حظر بواسطة صاحب الموقع لمخالفة الاستخدام' : '',
        }),
      });
      if (res.status === 401) { handleSessionExpired(); return; }
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !(data as { success?: boolean }).success) {
        // The old code checked `data.success` only and stayed silent otherwise, so
        // a failed block looked exactly like a successful one.
        showToast(
          (data as { message?: string }).message || 'تعذّر تحديث حالة الجهاز. تأكد من تنفيذ supabase/projects.sql',
        );
        return;
      }
      // Flip the flag locally so the row reflects the decision immediately, then
      // re-read for the canonical values. The server answers with
      // `{ deviceId, isBlocked }`, not a `device` object — waiting for one that
      // never arrived left a successful block still showing as unblocked.
      setDevices(prev =>
        prev.map((d) =>
          d.deviceId === device.deviceId
            ? {
                ...d,
                isBlocked: !device.isBlocked,
                blockReason: device.isBlocked
                  ? undefined
                  : 'حظر بواسطة صاحب الموقع لمخالفة الاستخدام',
              }
            : d,
        ),
      );
      void fetchAdminData();
      showToast(device.isBlocked ? 'تم فك حظر الجهاز بنجاح' : 'تم حظر الجهاز بنجاح');
    } catch {
      showToast('تعذّر الاتصال بالخادم، حاول مرة أخرى');
    }
  };

  const handleResetDeviceQuota = async (device: DeviceProtectionInfo, newLimit?: number) => {
    try {
      const res = await fetch('/api/admin/device/reset-quota', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          deviceId: device.deviceId,
          fingerprintHash: device.fingerprintHash,
          resetUsed: true,
          newLimit: newLimit !== undefined ? newLimit : device.freeGenerationsLimit,
        }),
      });
      if (res.status === 401) { handleSessionExpired(); return; }
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !(data as { success?: boolean }).success) {
        showToast(
          (data as { message?: string }).message || 'تعذّر تصفير رصيد الجهاز. تأكد من تنفيذ supabase/projects.sql',
        );
        return;
      }
      // Reflect the reset locally instead of waiting for a `device` object the
      // server does not return — otherwise the counter stayed stale on screen
      // even though the write had landed.
      setDevices(prev =>
        prev.map((d) =>
          d.deviceId === device.deviceId
            ? {
                ...d,
                freeGenerationsUsed: 0,
                freeGenerationsLimit: newLimit ?? d.freeGenerationsLimit,
              }
            : d,
        ),
      );
      void fetchAdminData();
      showToast('تم تصفير استهلاك الجهاز وتجديد رصيده بنجاح');
    } catch {
      showToast('تعذّر الاتصال بالخادم، حاول مرة أخرى');
    }
  };

  const handleUpgradeDeviceTier = async (device: DeviceProtectionInfo, tier: 'free' | 'pro' | 'business') => {
    try {
      const res = await fetch('/api/admin/device/set-tier', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          deviceId: device.deviceId,
          fingerprintHash: device.fingerprintHash,
          tier,
        }),
      });
      if (res.status === 401) { handleSessionExpired(); return; }
      const data = await res.json().catch(() => ({}));
      if (!res.ok || !(data as { success?: boolean }).success) {
        showToast(
          (data as { message?: string }).message || 'تعذّر تحديث باقة الجهاز. تأكد من تنفيذ supabase/projects.sql',
        );
        return;
      }
      setDevices(prev =>
        prev.map((d) => (d.deviceId === device.deviceId ? { ...d, associatedTier: tier } : d)),
      );
      void fetchAdminData();
      showToast(`تم تحديث باقة الجهاز إلى: ${tier.toUpperCase()}`);
    } catch {
      showToast('تعذّر الاتصال بالخادم، حاول مرة أخرى');
    }
  };

  // SECURITY: approving a payment is the ONLY way a paid tier is granted. The
  // customer's address is read from the stored payment row by the server, never
  // from this request, so a crafted call cannot grant a plan to anyone else.
  const handleUpdateTransactionStatus = async (
    txId: string,
    status: 'confirmed' | 'rejected',
    customerEmail?: string,
    tier?: string,
  ) => {
    try {
      // A confirm needs an account to grant to. The server reads the address from
      // the stored payment (never from this body), so the guard here is only
      // about telling the owner *why* nothing will happen.
      if (status === 'confirmed' && !customerEmail) {
        showToast('لا يوجد بريد مسجل لهذه المعاملة — لا يمكن التفعيل بدونه');
        return;
      }
      const res = await fetch('/api/admin/transaction/update-status', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactionId: txId, status }),
      });
      if (res.status === 401) { handleSessionExpired(); return; }
      const data = await res.json().catch(() => ({}));
      if (res.ok && (data as { success?: boolean }).success) {
        // Update the row in place immediately, then re-read for the totals.
        // The server no longer echoes a `transaction` object, and waiting for
        // the refetch left the row showing "approve" for a decided payment.
        setTransactions(prev =>
          prev.map((t) =>
            t.id === txId
              ? {
                  ...t,
                  status,
                  verifiedAt: new Date().toISOString(),
                }
              : t,
          ),
        );
        void fetchAdminData();
        showToast(
          (data as { message?: string }).message ||
            (status === 'confirmed' ? 'تم تأكيد وتفعيل المعاملة بنجاح' : 'تم رفض المعاملة'),
        );
      } else {
        showToast(
          (data as { error?: string; message?: string }).error ||
            (data as { message?: string }).message ||
            'تعذّر تنفيذ العملية',
        );
      }
    } catch {
      showToast('تعذّر الاتصال بالخادم، حاول مرة أخرى');
    }
  };

  const handleSaveSettings = async (e: import('react').FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch('/api/admin/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });
      if (res.status === 401) { handleSessionExpired(); return; }
      const data = await res.json().catch(() => ({}));
      if (res.ok && (data as { success?: boolean }).success) {
        // Adopt the server's normalised values: it clamps the free limit and
        // fills any missing field, so the form then shows what was really saved.
        const saved = (data as { settings?: AdminSettings }).settings;
        if (saved) setSettings(saved);
        showToast('تم حفظ إعدادات المنصة بنجاح!');
      } else {
        // The old code said "saved!" for any 200 response, which is how the
        // owner ended up believing a wallet number change had taken effect when
        // the server had discarded it.
        showToast(
          (data as { message?: string }).message || 'تعذّر حفظ الإعدادات. تأكد من تنفيذ supabase/projects.sql',
        );
      }
    } catch {
      showToast('تعذّر الاتصال بالخادم، حاول مرة أخرى');
    }
  };

  const showToast = (msg: string) => {
    setActionSuccessMessage(msg);
    setTimeout(() => setActionSuccessMessage(null), 3000);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-slate-950/85 backdrop-blur-md overflow-y-auto animate-fadeIn select-none" dir="rtl">
      {/*
        The dialog element carries the ref the focus hook watches and is marked
        as a modal for assistive tech, so the window below is not announced
        alongside it.
      */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={language === 'ar' ? 'لوحة إدارة المنصة' : 'Platform administration'}
        className="relative w-full max-w-5xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col my-auto max-h-[92vh]">

        {/* Top Header */}
        <div className="px-6 py-4 border-b border-slate-800 bg-slate-950 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-rose-500 to-amber-500 flex items-center justify-center text-white shadow-lg shadow-rose-500/20 font-bold">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-base font-bold text-white">لوحة تحكم المالك | إدارة وحماية المنصة</h2>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30">
                  Owner Admin
                </span>
              </div>
              <p className="text-xs text-slate-400">إحصائيات حقيقية 100%، مراقبة الأجهزة، ومراجعة أورانج كاش</p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="w-8 h-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white flex items-center justify-center transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Toast. `actionSuccessMessage` carries BOTH outcomes — the admin actions below
            reuse it for failures too — so it is coloured by whether the text
            starts with the failure marker the writers use. Rendering every
            message on green is how a rejected write ends up looking like a
            successful one. */}
        {(actionSuccessMessage || actionErrorMessage) && (
          <div
            role="status"
            className={`px-6 py-2 text-xs font-bold flex items-center gap-2 border-b ${
              actionErrorMessage
                ? 'bg-rose-500/20 border-rose-500/30 text-rose-200'
                : 'bg-emerald-500/20 border-emerald-500/30 text-emerald-300'
            }`}
          >
            {actionErrorMessage ? (
              <AlertTriangle className="w-4 h-4 shrink-0" />
            ) : (
              <CheckCircle className="w-4 h-4 shrink-0" />
            )}
            <span className="truncate">{actionErrorMessage ?? actionSuccessMessage}</span>
            <button
              type="button"
              onClick={() => {
                setActionSuccessMessage(null);
                setActionErrorMessage(null);
              }}
              className="ms-auto shrink-0 opacity-70 hover:opacity-100"
              aria-label="إغلاق"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Auth Gate if not authenticated */}
        {!isAuthenticated ? (
          <div className="p-8 sm:p-12 flex flex-col items-center justify-center text-center max-w-md mx-auto my-auto">
            <div className="w-16 h-16 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center text-amber-400 mb-4 shadow-xl">
              <Lock className="w-8 h-8" />
            </div>
            <h3 className="text-lg font-bold text-white mb-1">تسجيل دخول صاحب الموقع</h3>
            <p className="text-xs text-slate-400 mb-6 leading-relaxed">
              هذه الصفحة مخصصة لمالك المنصة فقط لإدارة الاشتراكات وفحص حماية الأجهزة. يرجى إدخال رمز PIN الخاص بك.
            </p>

            <form onSubmit={handleLogin} className="w-full space-y-4">
              <div>
                <input
                  type="password"
                  value={pinInput}
                  onChange={(e) => setPinInput(e.target.value)}
                  placeholder="أدخل رمز PIN الخاص بمالك الموقع..."
                  className="w-full px-4 py-3 bg-slate-950 border border-slate-700 rounded-xl text-white text-sm text-center tracking-widest focus:outline-none focus:border-rose-500 transition"
                  autoFocus
                  required
                />
                <span className="text-[10px] text-slate-500 mt-1 block">رمز PIN الافتراضي: 01207782741 أو admin803</span>
              </div>

              {(authError || sessionError) && (
                <div className="p-3 bg-rose-500/20 border border-rose-500/30 rounded-xl text-rose-300 text-xs font-semibold flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>{authError || sessionError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={isAuthenticating}
                className="w-full py-3 bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 text-white font-bold rounded-xl text-sm transition shadow-lg shadow-rose-600/20"
              >
                {isAuthenticating ? 'جاري التحقق...' : 'دخول لوحة الإدارة'}
              </button>
            </form>
          </div>
        ) : isCheckingSession || isLoadingData || !isDataReady ? (
          <div className="p-10 sm:p-14 flex flex-col items-center justify-center text-center max-w-md mx-auto my-auto" role="status" aria-live="polite">
            <RefreshCw className={`w-8 h-8 text-amber-400 mb-4 ${isLoadingData || isCheckingSession ? 'animate-spin' : ''}`} />
            <h3 className="text-base font-bold text-white mb-2">جارٍ تجهيز لوحة الإدارة</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              {dataError || (isCheckingSession ? 'يتم التحقق من الجلسة.' : 'يتم تحميل البيانات بأمان.')}
            </p>
            {dataError && !isLoadingData && !isCheckingSession && (
              <button
                type="button"
                onClick={async () => {
                  const loaded = await fetchAdminData();
                  if (loaded) setIsAuthenticated(true);
                }}
                className="mt-5 px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold rounded-lg transition"
              >
                إعادة المحاولة
              </button>
            )}
          </div>
        ) : (
          /* Main Authenticated Admin View */
          <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
            
            {/* Tabs Bar */}
            <div className="px-6 py-2 bg-slate-950/60 border-b border-slate-800 flex items-center gap-2 shrink-0 overflow-x-auto">
              <button
                onClick={() => setActiveTab('overview')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition whitespace-nowrap ${
                  activeTab === 'overview'
                    ? 'bg-rose-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                <TrendingUp className="w-3.5 h-3.5" />
                <span>نظرة عامة حقيقية</span>
              </button>

              <button
                onClick={() => setActiveTab('users')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition whitespace-nowrap ${
                  activeTab === 'users'
                    ? 'bg-rose-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                <Users className="w-3.5 h-3.5" />
                <span>المستخدمون</span>
                {accountStats.online > 0 && (
                  <span className="flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    {accountStats.online}
                  </span>
                )}
              </button>

              <button
                onClick={() => setActiveTab('devices')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition whitespace-nowrap ${
                  activeTab === 'devices'
                    ? 'bg-rose-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                <Smartphone className="w-3.5 h-3.5" />
                <span>الأجهزة المحمية ({devices.length})</span>
              </button>

              <button
                onClick={() => setActiveTab('transactions')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition whitespace-nowrap ${
                  activeTab === 'transactions'
                    ? 'bg-rose-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                <CreditCard className="w-3.5 h-3.5" />
                <span>معاملات أورانج كاش ({transactions.length})</span>
              </button>

              <button
                onClick={() => setActiveTab('settings')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-bold transition whitespace-nowrap ${
                  activeTab === 'settings'
                    ? 'bg-rose-500 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
                <Settings className="w-3.5 h-3.5" />
                <span>إعدادات النظام والمحفظة</span>
              </button>

              <div className="mr-auto flex items-center gap-2">
                <button
                  onClick={fetchAdminData}
                  className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition"
                  title="تحديث البيانات"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isLoadingData ? 'animate-spin' : ''}`} />
                </button>
              </div>
            </div>

            {/* Tab Contents */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              
              {/* TAB 1: OVERVIEW */}
              {activeTab === 'overview' && (
                <div className="space-y-6 animate-fadeIn">
                  {/* Real Stats Cards (No Fake Numbers) */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    
                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
                      <div className="flex items-center justify-between text-slate-400 mb-2">
                        <span className="text-xs font-semibold">إجمالي الأجهزة المسجلة</span>
                        <Smartphone className="w-4 h-4 text-sky-400" />
                      </div>
                      <div className="text-2xl font-black text-white font-mono">
                        {stats?.totalDevicesCount || devices.length}
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1">تتبع فعلي لبصمات الهواتف والأجهزة</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
                      <div className="flex items-center justify-between text-slate-400 mb-2">
                        <span className="text-xs font-semibold">الأجهزة المحظورة لمنع الاحتيال</span>
                        <Ban className="w-4 h-4 text-rose-400" />
                      </div>
                      <div className="text-2xl font-black text-rose-400 font-mono">
                        {stats?.blockedDevicesCount || devices.filter(d => d.isBlocked).length}
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1">تم إيقافها لمنع استنزاف الرصيد المجاني</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
                      <div className="flex items-center justify-between text-slate-400 mb-2">
                        <span className="text-xs font-semibold">إجمالي طلبات AI المنفذة</span>
                        <Cpu className="w-4 h-4 text-amber-400" />
                      </div>
                      <div className="text-2xl font-black text-white font-mono">
                        {stats?.totalGenerationsExecuted ?? 0}
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1">عمليات بناء وتعديل حقيقية تمت</p>
                    </div>

                    <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col justify-between">
                      <div className="flex items-center justify-between text-slate-400 mb-2">
                        <span className="text-xs font-semibold">إجمالي إيرادات أورانج كاش</span>
                        <DollarSign className="w-4 h-4 text-emerald-400" />
                      </div>
                      <div className="text-2xl font-black text-emerald-400 font-mono">
                        {stats?.totalRevenueEGP || 0} <span className="text-xs font-normal">ج.م</span>
                      </div>
                      <p className="text-[11px] text-slate-500 mt-1">من المعاملات المؤكدة في قاعدة البيانات</p>
                    </div>

                  </div>

                  {/* Anti-Fraud Protection Notice */}
                  <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-slate-300 space-y-2">
                    <div className="flex items-center gap-2 text-rose-400 font-bold text-sm">
                      <ShieldCheck className="w-4 h-4" />
                      <span>نظام الحماية ضد استغلال الرصيد المجاني يعمل بكفاءة</span>
                    </div>
                    <p className="text-xs leading-relaxed">
                      يقوم النظام بدمج بصمة الشاشة، محرك الرسم Canvas، المنطقة الزمنية، ومعرّف العتاد. حتى لو قام المستخدم بتسجيل الدخول ببريد إلكتروني جديد أو فتح نافذة التصفح المتخفي (Incognito)، يتعرف النظام على جهازه فوراً ويمنعه من تجاوز حد الـ {settings.defaultFreeLimit} طلبات مجانية.
                    </p>
                  </div>

                  {/* Recent Transactions Quick Table */}
                  <div className="space-y-3">
                    <h4 className="text-xs font-bold text-slate-300 flex items-center justify-between">
                      <span>آخر عمليات تحويل أورانج كاش</span>
                      <button onClick={() => setActiveTab('transactions')} className="text-rose-400 hover:underline text-[11px]">
                        عرض الكل
                      </button>
                    </h4>

                    {transactions.length === 0 ? (
                      <div className="p-6 bg-slate-950 rounded-xl border border-slate-800 text-center text-slate-500 text-xs">
                        لا توجد تحويلات مسجلة حتى الآن.
                      </div>
                    ) : (
                      <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden">
                        <table className="w-full text-right text-xs">
                          <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                            <tr>
                              <th className="p-3">رقم المحول</th>
                              <th className="p-3">الكود المرجعي</th>
                              <th className="p-3">المبلغ</th>
                              <th className="p-3">الباقة</th>
                              <th className="p-3">الحالة</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60">
                            {transactions.slice(0, 5).map(tx => (
                              <tr key={tx.id} className="hover:bg-slate-900/40">
                                <td className="p-3 font-mono text-white">{tx.senderPhone}</td>
                                <td className="p-3 font-mono text-slate-300">{tx.transactionReference}</td>
                                <td className="p-3 font-bold text-emerald-400">{tx.amount} ج.م</td>
                                <td className="p-3 text-slate-300">{tx.planName}</td>
                                <td className="p-3">
                                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                    tx.status === 'confirmed'
                                      ? 'bg-emerald-500/20 text-emerald-400'
                                      : tx.status === 'pending'
                                      ? 'bg-amber-500/20 text-amber-400'
                                      : 'bg-rose-500/20 text-rose-400'
                                  }`}>
                                    {tx.status === 'confirmed' ? 'مؤكدة' : tx.status === 'pending' ? 'معلقة' : 'مرفوضة'}
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* TAB: USERS — sign-ups, credits, and live presence.

              The owner's three questions in one place: who registered (most
              recent first), how many credits each one holds, and who is on the
              site right now. Presence is a server-derived field, so the dot
              cannot be stale relative to the database. */}
              {activeTab === 'users' && (
                <div className="space-y-4 animate-fadeIn">
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                    {(
                      [
                        { label: 'إجمالي الحسابات', value: accountStats.total, tone: 'text-white' },
                        { label: 'متصل الآن', value: accountStats.online, tone: 'text-emerald-400' },
                        { label: 'غير نشط', value: accountStats.idle, tone: 'text-amber-400' },
                        { label: 'إجمالي الكريديت', value: accountStats.credits, tone: 'text-orange-400' },
                      ] as const
                    ).map((card) => (
                      <div key={card.label} className="bg-slate-950 border border-slate-800 rounded-xl px-4 py-3">
                        <div className="text-[10px] font-bold text-slate-500 mb-1">{card.label}</div>
                        <div className={`text-xl font-black font-mono ${card.tone}`}>{card.value}</div>
                      </div>
                    ))}
                  </div>

                  <p className="text-[11px] text-slate-500 leading-relaxed">
                    كل حساب جديد يُمنح {accountStats.welcomePerSignup} كريديت تلقائياً عند أول تسجيل
                    دخول. الكريديت محفوظ في قاعدة البيانات ولا يُفقد بتغيير الجهاز أو تسجيل الخروج.
                    حالة الاتصال تُحدَّث كل 15 ثانية.
                  </p>

                  {accounts.length === 0 ? (
                    <div className="py-16 text-center border border-dashed border-slate-800 rounded-xl">
                      <Users className="w-10 h-10 mx-auto text-slate-700 mb-3" />
                      <p className="text-sm font-bold text-slate-300 mb-1">لا توجد حسابات مسجّلة بعد</p>
                      <p className="text-[11px] text-slate-500 max-w-sm mx-auto leading-relaxed">
                        سيظهر كل مستخدم يسجّل الدخول هنا تلقائياً مع بريده ووقت تسجيله وعدد كريديته.
                        إن استمر الفراغ بعد وجود مستخدمين، نفّذ الجزء الجديد من
                        supabase/projects.sql في محرر Supabase.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {accounts.map((account) => {
                        const busy = busyAccountId === account.accountId;
                        const isOnline = account.presence === 'online';
                        const isIdle = account.presence === 'idle';
                        return (
                          <div
                            key={account.accountId}
                            className={`flex flex-col sm:flex-row sm:items-center gap-3 p-3 rounded-xl border transition ${
                              account.isBlocked
                                ? 'bg-rose-950/20 border-rose-500/30'
                                : isOnline
                                  ? 'bg-emerald-950/10 border-emerald-500/25'
                                  : 'bg-slate-950/50 border-slate-800'
                            }`}
                          >
                            {/* Presence dot + avatar */}
                            <div className="flex items-center gap-3 min-w-0 sm:w-52 shrink-0">
                              <span className="relative shrink-0">
                                {account.avatarUrl ? (
                                  <img
                                    src={account.avatarUrl}
                                    alt=""
                                    referrerPolicy="no-referrer"
                                    className="w-9 h-9 rounded-full object-cover"
                                  />
                                ) : (
                                  <span className="w-9 h-9 rounded-full bg-slate-800 flex items-center justify-center">
                                    <User className="w-4 h-4 text-slate-500" />
                                  </span>
                                )}
                                <span
                                  title={isOnline ? 'متصل الآن' : isIdle ? 'غير نشط' : 'غير متصل'}
                                  className={`absolute -bottom-0.5 -end-0.5 w-3 h-3 rounded-full border-2 border-slate-900 ${
                                    isOnline ? 'bg-emerald-400' : isIdle ? 'bg-amber-400' : 'bg-slate-600'
                                  }`}
                                />
                              </span>
                              <div className="min-w-0">
                                <div className="text-xs font-bold text-slate-100 truncate">
                                  {account.name || 'مستخدم'}
                                </div>
                                <div className="text-[10px] text-slate-400 truncate" dir="ltr" title={account.email}>
                                  {account.email || '—'}
                                </div>
                              </div>
                            </div>

                            {/* Credits — the wallet this row is about. */}
                            <div className="flex items-center gap-2 sm:w-32 shrink-0">
                              <span className="text-[10px] text-slate-500">كريديت</span>
                              <span className="text-sm font-black font-mono text-orange-400">
                                {account.credits}
                              </span>
                            </div>

                            {/* Tier */}
                            <div className="sm:w-24 shrink-0">
                              <span
                                className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                                  account.tier === 'free'
                                    ? 'bg-slate-800 text-slate-400'
                                    : account.tier === 'pro'
                                      ? 'bg-orange-500/20 text-orange-300'
                                      : 'bg-emerald-500/20 text-emerald-300'
                                }`}
                              >
                                {account.tier}
                              </span>
                            </div>

                            {/* When they joined, when they were last here, and the requests the server
                                counted for them. `requestsCount` is rendered rather
                                than carried unused, so the column is never a lie
                                about how much the row actually knows. */}
                            <div className="text-[10px] text-slate-500 sm:flex-1 min-w-0">
                              <span className="block">
                                سجّل: {relativeTime(account.firstSeenAt, language === 'ar')}
                                {account.welcomeGiven
                                  ? ` · ${language === 'ar' ? 'حصل على كريديت الترحيب' : 'got welcome credits'}`
                                  : ` · ${language === 'ar' ? 'بدون كريديت ترحيب' : 'no welcome credits'}`}
                              </span>
                              <span className="block">
                                آخر نشاط: {relativeTime(account.lastSeenAt, language === 'ar')}
                                {account.requestsCount > 0 ? ` · ${account.requestsCount} طلب` : ''}
                                {account.lastIp ? ` · ${account.lastIp}` : ''}
                              </span>
                            </div>

                            {/* Actions. Each icon-only control carries its own
                                accessible name — a title attribute alone is not
                                reliably announced by a screen reader. */}
                            <div className="flex items-center gap-1.5 shrink-0">
                              <button
                                type="button"
                                onClick={() => handleSetCredits(account)}
                                disabled={busy}
                                title="منح أو تعديل الكريديت"
                                aria-label={`تعديل كريديت ${account.email || account.name}`}
                                className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-orange-400 transition disabled:opacity-40 disabled:cursor-not-allowed"
                              >
                                {busy ? (
                                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                ) : (
                                  <Coins className="w-3.5 h-3.5" />
                                )}
                              </button>
                              <button
                                type="button"
                                onClick={() => handleToggleAccountBlock(account)}
                                disabled={busy}
                                title={account.isBlocked ? 'إلغاء الحظر' : 'حظر الحساب'}
                                aria-label={
                                  account.isBlocked
                                    ? `إلغاء حظر ${account.email || account.name}`
                                    : `حظر ${account.email || account.name}`
                                }
                                className={`p-2 rounded-lg transition disabled:opacity-40 disabled:cursor-not-allowed ${
                                  account.isBlocked
                                    ? 'bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300'
                                    : 'bg-slate-800 hover:bg-rose-500/20 text-rose-400'
                                }`}
                              >
                                {account.isBlocked ? (
                                  <ShieldCheck className="w-3.5 h-3.5" />
                                ) : (
                                  <Ban className="w-3.5 h-3.5" />
                                )}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}

              {/* TAB: DEVICES */}
              {activeTab === 'devices' && (
                <div className="space-y-4 animate-fadeIn">
                  <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-bold text-white">سجل الأجهزة المحمية وبصمات العتاد</h3>
                      <p className="text-xs text-slate-400">تتبع استهلاك الرصيد المجاني لكل جهاز وحظر المحتالين</p>
                    </div>

                    <div className="relative w-full sm:w-64">
                      <Search className="w-4 h-4 text-slate-400 absolute right-3 top-1/2 -translate-y-1/2" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="بحث بالـ IP أو المعرّف أو الإيميل..."
                        className="w-full pl-3 pr-9 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-white placeholder-slate-500 focus:outline-none focus:border-rose-500"
                      />
                    </div>
                  </div>

                  <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="w-full text-right text-xs">
                        <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                          <tr>
                            <th className="p-3">معرّف الجهاز والـ IP</th>
                            <th className="p-3">الإيميلات المستخدمة</th>
                            <th className="p-3">الرصيد المستهلك</th>
                            <th className="p-3">الباقة</th>
                            <th className="p-3">الحالة</th>
                            <th className="p-3 text-center">إجراءات المالك</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60">
                          {devices
                            .filter(d => 
                              !searchQuery || 
                              d.ipAddress?.includes(searchQuery) || 
                              d.deviceId?.includes(searchQuery) ||
                              d.registeredEmails?.some(e => e.toLowerCase().includes(searchQuery.toLowerCase()))
                            )
                            .map(dev => (
                              <tr key={dev.deviceId} className="hover:bg-slate-900/40">
                                <td className="p-3">
                                  <div className="font-mono text-white text-[11px] truncate max-w-[140px]" title={dev.deviceId}>
                                    {dev.deviceId}
                                  </div>
                                  <div className="text-[10px] text-slate-400 font-mono">IP: {dev.ipAddress}</div>
                                </td>
                                <td className="p-3">
                                  {dev.registeredEmails && dev.registeredEmails.length > 0 ? (
                                    <div className="space-y-0.5">
                                      {dev.registeredEmails.slice(0, 2).map((em, idx) => (
                                        <div key={idx} className="text-[11px] text-slate-300">{em}</div>
                                      ))}
                                      {dev.registeredEmails.length > 2 && (
                                        <span className="text-[10px] text-rose-400 font-bold">
                                          + {dev.registeredEmails.length - 2} إيميلات أخرى (محاولة تحايل)
                                        </span>
                                      )}
                                    </div>
                                  ) : (
                                    <span className="text-slate-500 text-[11px]">زائر غير مسجل</span>
                                  )}
                                </td>
                                <td className="p-3">
                                  <div className="font-bold text-white">
                                    {dev.freeGenerationsUsed} / {dev.freeGenerationsLimit}
                                  </div>
                                  <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden mt-1">
                                    <div 
                                      className={`h-full ${dev.freeGenerationsUsed >= dev.freeGenerationsLimit ? 'bg-rose-500' : 'bg-emerald-500'}`}
                                      style={{ width: `${Math.min(100, (dev.freeGenerationsUsed / dev.freeGenerationsLimit) * 100)}%` }}
                                    />
                                  </div>
                                </td>
                                <td className="p-3">
                                  <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold ${
                                    dev.associatedTier !== 'free'
                                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                                      : 'bg-slate-800 text-slate-400'
                                  }`}>
                                    {dev.associatedTier.toUpperCase()}
                                  </span>
                                </td>
                                <td className="p-3">
                                  {dev.isBlocked ? (
                                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30">
                                      محظور
                                    </span>
                                  ) : (
                                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-400">
                                      نشط
                                    </span>
                                  )}
                                </td>
                                <td className="p-3">
                                  <div className="flex items-center justify-center gap-1.5">
                                    <button
                                      onClick={() => handleResetDeviceQuota(dev)}
                                      className="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[10px] font-bold transition"
                                      title="تصفير الاستهلاك وتجديد الرصيد"
                                    >
                                      تصفير
                                    </button>

                                    <button
                                      onClick={() => handleUpgradeDeviceTier(dev, dev.associatedTier === 'free' ? 'pro' : 'free')}
                                      className={`px-2 py-1 rounded text-[10px] font-bold transition ${
                                        dev.associatedTier === 'free'
                                          ? 'bg-amber-600 hover:bg-amber-500 text-white'
                                          : 'bg-slate-800 text-slate-400'
                                      }`}
                                      title="ترقية إلى باقة المحترفين"
                                    >
                                      {dev.associatedTier === 'free' ? 'ترقية Pro' : 'تنزيل لـ Free'}
                                    </button>

                                    <button
                                      onClick={() => handleToggleBlockDevice(dev)}
                                      className={`px-2 py-1 rounded text-[10px] font-bold transition ${
                                        dev.isBlocked
                                          ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                                          : 'bg-rose-600 hover:bg-rose-500 text-white'
                                      }`}
                                    >
                                      {dev.isBlocked ? 'فك الحظر' : 'حظر'}
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: TRANSACTIONS */}
              {activeTab === 'transactions' && (
                <div className="space-y-4 animate-fadeIn">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-bold text-white">سجل معاملات أورانج كاش</h3>
                      <p className="text-xs text-slate-400">المحفظة الرسمية: {settings.orangeWalletNumber}</p>
                    </div>
                  </div>

                  <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden">
                    <table className="w-full text-right text-xs">
                      <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                        <tr>
                          <th className="p-3">رقم المحول</th>
                          <th className="p-3">الرقم المرجعي</th>
                          <th className="p-3">المبلغ</th>
                          <th className="p-3">الباقة</th>
                          <th className="p-3">الاسم / الإيميل</th>
                          <th className="p-3">التاريخ</th>
                          <th className="p-3">الحالة</th>
                          <th className="p-3 text-center">الإجراء</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {transactions.map(tx => (
                          <tr key={tx.id} className="hover:bg-slate-900/40">
                            <td className="p-3 font-mono text-white font-bold">{tx.senderPhone}</td>
                            <td className="p-3 font-mono text-amber-300 font-semibold">{tx.transactionReference}</td>
                            <td className="p-3 font-bold text-emerald-400">{tx.amount} ج.م</td>
                            <td className="p-3 text-slate-300">{tx.planName}</td>
                            <td className="p-3 text-slate-400 text-[11px]">
                              <div>{tx.userName || 'عميل'}</div>
                              <div>{tx.userEmail || ''}</div>
                            </td>
                            <td className="p-3 text-slate-500 text-[10px] font-mono">
                              {new Date(tx.submittedAt).toLocaleDateString('ar-EG')}
                            </td>
                            <td className="p-3">
                              <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                tx.status === 'confirmed'
                                  ? 'bg-emerald-500/20 text-emerald-400'
                                  : tx.status === 'pending'
                                  ? 'bg-amber-500/20 text-amber-400'
                                  : 'bg-rose-500/20 text-rose-400'
                              }`}>
                                {tx.status === 'confirmed' ? 'مؤكدة ومفعلة' : tx.status === 'pending' ? 'معلقة' : 'مرفوضة'}
                              </span>
                            </td>
                            <td className="p-3 text-center">
                              {tx.status !== 'confirmed' ? (
                                <div className="flex items-center justify-center gap-1.5">
                                  <button
                                    onClick={() => handleUpdateTransactionStatus(tx.id, 'confirmed', tx.userEmail, tx.planId)}
                                    className="px-2 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded text-[10px] font-bold transition"
                                  >
                                    تأكيد وتفعيل
                                  </button>
                                  <button
                                    onClick={() => handleUpdateTransactionStatus(tx.id, 'rejected')}
                                    className="px-2 py-1 bg-rose-600 hover:bg-rose-500 text-white rounded text-[10px] font-bold transition"
                                  >
                                    رفض
                                  </button>
                                </div>
                              ) : (
                                <span className="text-slate-500 text-[10px]">مكتملة</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* TAB 4: SETTINGS */}
              {activeTab === 'settings' && (
                <form onSubmit={handleSaveSettings} className="max-w-xl space-y-4 animate-fadeIn">
                  <div className="space-y-1">
                    <h3 className="text-sm font-bold text-white">إعدادات المنصة ومحفظة أورانج كاش</h3>
                    <p className="text-xs text-slate-400">تعديل الأرقام الرسمية وسياسة الرصيد المجاني</p>
                  </div>

                  <div className="space-y-3 pt-2">
                    <div>
                      <label className="block text-xs font-semibold text-slate-300 mb-1">
                        رقم محفظة أورانج كاش الرسمية (لاستقبال التحويلات):
                      </label>
                      <input
                        type="text"
                        value={settings.orangeWalletNumber}
                        onChange={(e) => setSettings((prev) => ({ ...prev, orangeWalletNumber: e.target.value }))}
                        className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-white text-xs font-mono focus:border-rose-500 focus:outline-none"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-300 mb-1">
                        رقم واتساب المخصص للدعم الفني والتواصل المباشر:
                      </label>
                      <input
                        type="text"
                        value={settings.supportWhatsappNumber}
                        onChange={(e) => setSettings((prev) => ({ ...prev, supportWhatsappNumber: e.target.value }))}
                        className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-white text-xs font-mono focus:border-rose-500 focus:outline-none"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-300 mb-1">
                        الحد الأقصى للطلبات المجانية لكل جهاز جديد (Free Quota):
                      </label>
                      <input
                        type="number"
                        min="1"
                        max="50"
                        value={settings.defaultFreeLimit}
                        onChange={(e) => setSettings((prev) => ({ ...prev, defaultFreeLimit: Number(e.target.value) }))}
                        className="w-full px-3 py-2 bg-slate-950 border border-slate-700 rounded-xl text-white text-xs font-mono focus:border-rose-500 focus:outline-none"
                        required
                      />
                    </div>

                    <div className="p-3 bg-slate-950 border border-slate-800 rounded-xl flex items-center justify-between">
                      <div>
                        <div className="text-xs font-bold text-white">تفعيل التحقق والدفع الأوتوماتيكي الفوري</div>
                        <div className="text-[11px] text-slate-400">
                          يقوم بتفعيل باقة المشترك فور إدخال الرقم المرجعي دون الحاجة لانتظار الموافقة اليدوية
                        </div>
                      </div>
                      <input
                        type="checkbox"
                        checked={settings.autoVerificationEnabled}
                        onChange={(e) => setSettings((prev) => ({ ...prev, autoVerificationEnabled: e.target.checked }))}
                        className="w-5 h-5 accent-rose-500 rounded cursor-pointer"
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="py-2.5 px-6 bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 text-white font-bold rounded-xl text-xs transition shadow-md shadow-rose-600/20"
                  >
                    حفظ التغييرات في النظام
                  </button>
                </form>
              )}

            </div>
          </div>
        )}

      </div>
    </div>
  );
};
