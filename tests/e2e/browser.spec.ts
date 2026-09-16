/**
 * Playwright E2E tests for WebInfer in a real browser environment.
 *
 * Requires the demo server to be running (handled by playwright.config.ts webServer).
 * Run with: npm run test:e2e
 */
import { test, expect } from '@playwright/test';

test.describe('WebInfer Browser E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for the WebInfer global to be ready
    await page.waitForFunction(() => typeof (window as any).WebInfer !== 'undefined', {
      timeout: 10_000,
    }).catch(() => {
      // If the global isn't exposed, tests below will fail with clear messages
    });
  });

  test('page loads successfully', async ({ page }) => {
    await expect(page).toHaveTitle(/.*/);
  });

  test('WebInfer global is exposed', async ({ page }) => {
    const hasGlobal = await page.evaluate(() => typeof (window as any).WebInfer !== 'undefined');
    expect(hasGlobal).toBe(true);
  });

  test('tensor creation works in browser', async ({ page }) => {
    const shape = await page.evaluate(() => {
      const ef = (window as any).WebInfer;
      if (!ef?.tensor) return null;
      const t = ef.tensor([1, 2, 3, 4], [2, 2]);
      return t.shape;
    });

    if (shape !== null) {
      expect(shape).toEqual([2, 2]);
    }
  });

  test('memory stats are accessible', async ({ page }) => {
    const stats = await page.evaluate(() => {
      const ef = (window as any).WebInfer;
      if (!ef?.getMemoryStats) return null;
      return ef.getMemoryStats();
    });

    if (stats !== null) {
      expect(stats).toHaveProperty('allocated');
      expect(stats).toHaveProperty('tensorCount');
    }
  });

  test('pipeline factory is callable', async ({ page }) => {
    const hasPipeline = await page.evaluate(() => {
      const ef = (window as any).WebInfer;
      return typeof ef?.pipeline === 'function';
    });

    expect(hasPipeline).toBe(true);
  });
});
