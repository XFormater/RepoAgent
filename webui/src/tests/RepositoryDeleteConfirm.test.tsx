import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { RepositoryDeleteConfirm } from "@/components/RepositoryDeleteConfirm";

const approval = {
  approval_id: "delete_123",
  repository_full_name: "owner/repo",
  commit_sha: "0123456789abcdef",
  source_path: "C:\\repos\\owner\\repo\\source",
  cache_path: "C:\\repos\\owner\\repo",
  expires_at: "2099-01-01T00:00:00Z",
};

describe("RepositoryDeleteConfirm", () => {
  it("shows the bound target and submits explicit approval", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();

    render(
      <RepositoryDeleteConfirm
        approval={approval}
        open
        onDismiss={vi.fn()}
        onReject={vi.fn()}
        onConfirm={onConfirm}
      />,
    );

    expect(screen.getByText("owner/repo")).toBeInTheDocument();
    expect(screen.getByText(approval.source_path)).toBeInTheDocument();
    expect(screen.getByText(approval.cache_path)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Delete local project" }));
    expect(onConfirm).toHaveBeenCalledOnce();
  });

  it("submits rejection without calling approval", async () => {
    const user = userEvent.setup();
    const onReject = vi.fn();
    const onConfirm = vi.fn();

    render(
      <RepositoryDeleteConfirm
        approval={approval}
        open
        onDismiss={vi.fn()}
        onReject={onReject}
        onConfirm={onConfirm}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Keep project" }));
    expect(onReject).toHaveBeenCalledOnce();
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
