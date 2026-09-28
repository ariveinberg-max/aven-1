import { expect, test } from "@playwright/test";

import { syntheticEdf } from "./edf";

test("overview shows the API online and links to the product pages", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "neurolayer" })).toBeVisible();
  await expect(page.getByText("API online")).toBeVisible();
  await page.getByRole("navigation").getByRole("link", { name: "Calibration game" }).click();
  await expect(page).toHaveURL(/\/calibrate$/);
});

test("calibration game: calibrate, play and see before/after accuracy", async ({ page }) => {
  await page.goto("/calibrate");
  await expect(page.getByLabel("Model")).toHaveValue(/nl-synthetic-demo@/);
  await page.getByLabel("Calibration trials per hand").selectOption("5");
  await page.getByLabel("Speed").selectOption("fast");
  await page.getByRole("button", { name: "Start calibration" }).click();

  await expect(page.getByRole("heading", { name: /Calibration · trial/ })).toBeVisible();
  await expect(page.getByText(/Imagine: (Left|Right) hand/)).toBeVisible();
  await expect(page.getByRole("heading", { name: /Play · trial/ })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByTestId("last-prediction")).toContainText(/Decoded (Left|Right) hand/);

  await expect(page.getByRole("heading", { name: "Result" })).toBeVisible({ timeout: 60_000 });
  const before = await page.getByTestId("baseline-accuracy").textContent();
  const after = await page.getByTestId("final-accuracy").textContent();
  expect(before).toMatch(/^\d+%$/);
  expect(after).toMatch(/^\d+%$/);
  console.log(`calibration game (synthetic): before ${before}, after ${after}`);
});

test("inspect: upload an EDF and see channels, quality flags and a preview", async ({ page }) => {
  await page.goto("/inspect");
  await page.getByLabel(/EEG recording/).setInputFiles({
    name: "session.edf",
    mimeType: "application/octet-stream",
    buffer: syntheticEdf(["C3", "Cz", "C4", "Pz"], 128, 12),
  });
  const summary = page.getByTestId("inspect-summary");
  await expect(summary).toContainText("EDF");
  await expect(summary).toContainText("128 Hz");
  await expect(summary).toContainText("12.0 s");
  await expect(page.getByRole("cell", { name: "flat" })).toBeVisible();
  await expect(page.getByRole("img", { name: /Signal preview: C3, Cz, C4, Pz/ })).toBeVisible();
});

test("inspect: rejects files that are not EDF/BDF", async ({ page }) => {
  await page.goto("/inspect");
  await page.getByLabel(/EEG recording/).setInputFiles({
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("hello"),
  });
  await expect(page.getByText(/Only EDF\/EDF\+/)).toBeVisible();
});
