export default {
  key: "ocr-livive-net",
  serve: { dir: "_site" },
  viewport: { width: 1280, height: 800 },
  durationMs: 30000,
  settleMs: 1200,
  trimIntro: true,

  async ready(page) {
    await page.waitForSelector("#open-train-modal-btn", { timeout: 30000 });
    await page.waitForTimeout(800);
  },

  // The page may boot with the training modal open and no models cached.
  // Train the 10,000-sample digits profile once, then close the modal so the
  // recorded tour shows populated classifier cards and a live consensus.
  async prelude(page) {
    await page
      .evaluate(() => document.querySelector("#modal-training-setup[open] .close-modal-btn")?.click())
      .catch(() => {});
    await page.waitForTimeout(300);

    await page.click("#open-train-modal-btn");
    await page.waitForSelector("#start-train-btn", { timeout: 10000 });
    await page.click('[data-size="10000"]');
    await page.waitForTimeout(300);

    await page.evaluate(() => {
      const btn = document.querySelector("#start-train-btn");
      if (btn && !btn.disabled) btn.click();
    });

    await page.waitForFunction(
      () => {
        const chip = document.querySelector("#active-profile-chip")?.textContent || "";
        const pred = document.querySelector("#pred-mlp")?.textContent?.trim() || "";
        const consensus = document.querySelector("#consensus-text")?.textContent || "";
        return (
          /10,000/.test(chip) &&
          pred !== "" &&
          pred !== "—" &&
          /agree/i.test(consensus)
        );
      },
      null,
      { timeout: 120000 },
    ).catch(() => {});

    await page.waitForTimeout(1200);
    await page
      .evaluate(() => document.querySelector("#modal-training-setup[open] .close-modal-btn")?.click())
      .catch(() => {});
    await page.waitForTimeout(500);

    // Pick a sample where all four classifiers agree, so the opening still is
    // the consensus state rather than a split decision.
    for (let i = 0; i < 15; i++) {
      await page.click("#random-test-btn").catch(() => {});
      await page.waitForTimeout(650);
      const consensus = await page.locator("#consensus-text").textContent().catch(() => "");
      if (/4\/4/.test(consensus || "")) break;
    }
    await page.waitForTimeout(500);
  },

  beats: [
    {
      label: "Four classifiers agree on a handwritten digit",
      action: { click: "#random-test-btn" },
      wait: 2000,
      shot: "trained-consensus",
      thumb: true,
    },
    {
      label: "Inspect the MLP arithmetic",
      action: [
        { scrollTo: 0 },
        { click: "#card-mlp .modal-trigger-btn" },
      ],
      wait: 1800,
      shot: "mlp-arithmetic",
    },
    {
      label: "Close the arithmetic inspector",
      action: {
        eval: () => {
          const modal = document.querySelector(".method-modal[open]");
          modal?.querySelector(".close-modal-btn")?.click();
        },
      },
      wait: 600,
    },
    {
      label: "Learning curve and sample efficiency",
      action: [
        { click: 'a[data-tab="experiments"]' },
        { scrollTo: 0 },
      ],
      wait: 2200,
      shot: "learning-curve",
    },
    {
      label: "Export the transparent ONNX graphs",
      action: { click: 'a[data-tab="export"]' },
      wait: 1800,
      shot: "onnx-export",
    },
    {
      label: "Back to the playground",
      action: [
        { click: 'a[data-tab="playground"]' },
        { scrollTo: 0 },
      ],
      wait: 1200,
    },
  ],

  thumbFrom: "trained-consensus",
};
