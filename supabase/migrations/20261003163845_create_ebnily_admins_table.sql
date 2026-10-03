/*
# Create ebnily_admins table

Stores a list of admin email addresses that the site owner can add or remove
from the dashboard. An account whose email matches a row in this table gets
full admin panel access (isOwner = true) WITHOUT needing the ADMIN_PIN —
they simply sign in with Google/GitHub and the admin panel is unlocked.

The original owner (SITE_OWNER_EMAIL) still needs the PIN as a second factor.
Added admins skip the PIN because the owner explicitly trusted their email.

1. New Tables
- ebnily_admins (id, email, added_by, added_at)

2. Security
- RLS enabled.
- anon and authenticated roles revoked (server uses service key only).
*/

CREATE TABLE IF NOT EXISTS public.ebnily_admins (
  id          text primary key,
  email       text        not null unique,
  added_by    text        not null default '',
  added_at    timestamptz not null default now()
);

ALTER TABLE public.ebnily_admins ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.ebnily_admins FROM anon, authenticated;