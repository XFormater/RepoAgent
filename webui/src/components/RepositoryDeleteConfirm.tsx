import { useEffect, useMemo, useState } from "react";
import { Clock3, FolderX, GitCommitHorizontal, ShieldAlert, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import type { RepositoryDeleteApprovalData } from "@/lib/types";

interface RepositoryDeleteConfirmProps {
  approval: RepositoryDeleteApprovalData;
  open: boolean;
  busy?: boolean;
  onDismiss: () => void;
  onReject: () => void;
  onConfirm: () => void;
}

export function RepositoryDeleteConfirm({
  approval,
  open,
  busy = false,
  onDismiss,
  onReject,
  onConfirm,
}: RepositoryDeleteConfirmProps) {
  const { t, i18n } = useTranslation();
  const expiresAt = useMemo(
    () => Date.parse(approval.expires_at),
    [approval.expires_at],
  );
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!open) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [open]);

  const expired = !Number.isFinite(expiresAt) || expiresAt <= now;
  const remainingSeconds = expired ? 0 : Math.ceil((expiresAt - now) / 1_000);
  const expiryLabel = Number.isFinite(expiresAt)
    ? new Intl.DateTimeFormat(i18n.resolvedLanguage || "zh-CN", {
        dateStyle: "short",
        timeStyle: "medium",
      }).format(new Date(expiresAt))
    : approval.expires_at;

  return (
    <AlertDialog open={open} onOpenChange={(next) => (!next ? onDismiss() : undefined)}>
      <AlertDialogContent className="w-[min(calc(100vw-2rem),34rem)] gap-0 overflow-hidden rounded-[28px] border border-destructive/20 bg-card/95 p-0 shadow-[0_28px_90px_rgba(15,23,42,0.28)] backdrop-blur-xl">
        <div className="border-b border-destructive/10 bg-gradient-to-br from-destructive/[0.12] via-destructive/[0.04] to-transparent px-6 pb-5 pt-6">
          <AlertDialogHeader className="space-y-0 text-left">
            <div className="mb-4 flex items-center gap-3">
              <div className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-destructive text-destructive-foreground shadow-[0_10px_28px_rgba(239,68,68,0.28)]">
                <Trash2 className="h-5 w-5" aria-hidden="true" />
              </div>
              <div className="min-w-0">
                <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-destructive">
                  <ShieldAlert className="h-3.5 w-3.5" aria-hidden="true" />
                  {t("repositoryDeleteApproval.riskLevel")}
                </div>
                <AlertDialogTitle className="truncate text-[21px] leading-tight tracking-[-0.025em]">
                  {t("repositoryDeleteApproval.title")}
                </AlertDialogTitle>
              </div>
            </div>
            <AlertDialogDescription className="text-[14px] leading-6 text-muted-foreground">
              {t("repositoryDeleteApproval.description")}
            </AlertDialogDescription>
          </AlertDialogHeader>
        </div>

        <div className="space-y-4 px-6 py-5">
          <div className="rounded-2xl border border-border/70 bg-muted/35 p-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
              {t("repositoryDeleteApproval.repository")}
            </p>
            <p className="mt-1.5 break-all text-[16px] font-semibold text-foreground">
              {approval.repository_full_name}
            </p>
            <div className="mt-3 flex items-center gap-2 text-[12px] text-muted-foreground">
              <GitCommitHorizontal className="h-4 w-4 shrink-0" aria-hidden="true" />
              <code className="truncate font-mono">{approval.commit_sha}</code>
            </div>
          </div>

          <div className="space-y-3 text-[12px]">
            <DetailRow
              icon={<FolderX className="h-4 w-4" aria-hidden="true" />}
              label={t("repositoryDeleteApproval.sourcePath")}
              value={approval.source_path}
            />
            <DetailRow
              icon={<Trash2 className="h-4 w-4" aria-hidden="true" />}
              label={t("repositoryDeleteApproval.deleteScope")}
              value={approval.cache_path}
              destructive
            />
            <DetailRow
              icon={<Clock3 className="h-4 w-4" aria-hidden="true" />}
              label={t("repositoryDeleteApproval.expiresAt")}
              value={
                expired
                  ? t("repositoryDeleteApproval.expired")
                  : t("repositoryDeleteApproval.expiresValue", {
                      time: expiryLabel,
                      seconds: remainingSeconds,
                    })
              }
              destructive={expired}
            />
          </div>

          <div className="rounded-2xl bg-destructive/[0.07] px-4 py-3 text-[12.5px] leading-5 text-foreground">
            <p className="font-semibold text-destructive">
              {t("repositoryDeleteApproval.warningTitle")}
            </p>
            <p className="mt-1 text-muted-foreground">
              {t("repositoryDeleteApproval.warningBody")}
            </p>
          </div>
        </div>

        <AlertDialogFooter className="!grid grid-cols-1 gap-3 border-t border-border/60 bg-muted/20 px-6 py-5 sm:grid-cols-2 sm:space-x-0">
          <Button
            type="button"
            variant="secondary"
            className="h-11 rounded-full text-[14px] font-semibold"
            disabled={busy}
            onClick={onReject}
          >
            {t("repositoryDeleteApproval.keep")}
          </Button>
          <Button
            type="button"
            variant="destructive"
            className="h-11 rounded-full text-[14px] font-semibold shadow-[0_10px_25px_rgba(239,68,68,0.24)]"
            disabled={busy || expired}
            onClick={onConfirm}
          >
            {busy
              ? t("repositoryDeleteApproval.submitting")
              : expired
                ? t("repositoryDeleteApproval.expiredAction")
                : t("repositoryDeleteApproval.confirm")}
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function DetailRow({
  icon,
  label,
  value,
  destructive = false,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  destructive?: boolean;
}) {
  return (
    <div className="flex items-start gap-3">
      <span
        className={destructive ? "mt-0.5 text-destructive" : "mt-0.5 text-muted-foreground"}
      >
        {icon}
      </span>
      <div className="min-w-0">
        <p className="font-medium text-muted-foreground">{label}</p>
        <p className="mt-0.5 break-all font-mono leading-5 text-foreground">{value}</p>
      </div>
    </div>
  );
}
