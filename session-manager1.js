// session-manager.js - Handles Stagehand initialization and session management
const { Stagehand } = require("@browserbasehq/stagehand");
const { z } = require("zod");
const { exec } = require("node:child_process");
// Zod schemas for configuration validation
const StagehandConfigSchema = z.object({
  mode: z.enum(["api", "experimental"]).default("experimental"),
  viewport: z
    .object({
      width: z.number().default(3024),
      height: z.number().default(1964),
    })
    .optional(),
  modelName: z.string().default("gpt-4.1"),
  modelClientOptions: z
    .object({
      apiKey: z.string().default(process.env.MODEL_API_KEY || process.env.OPENAI_API_KEY || ""),
    })
    .default({ apiKey: process.env.MODEL_API_KEY || process.env.OPENAI_API_KEY || "" }),
  verbose: z.number().min(0).max(3).default(3),
  headless: z.boolean().default(false),
  enableCaching: z.boolean().default(true),
  selfHeal: z.boolean().default(true),
  waitForCaptchaSolves: z.boolean().default(true),
});

// Type guard for error handling (inspired by LangChain)
function isErrorWithMessage(error) {
  return typeof error === "object" && error !== null && "message" in error && typeof error.message === "string";
}

class StagehandSessionManager {
  constructor() {
    this.stagehand = null;
    this.page = null;
    this.sessionId = null;
    this.isInitialized = false;
    this.isInUserInput = false;
    this.userInputResolver = null;
    this.currentMode = null; // 'api' or 'experimental'
    this.viewport = { width: 3024, height: 1964 }; // XDR Display P3 1600 aspect ratio
    this.setupSignalHandlers();
  }

  async getBravePath() {
    console.log("🔍 Detecting Brave browser path...");
    const os = await import("node:os");
    const platform = os.platform();

    let path;
    if (platform === "darwin") {
      path = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser";
    } else if (platform === "win32") {
      path = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe";
    } else {
      path = "/usr/bin/brave-browser";
    }

    console.log(`📍 Brave path detected: ${path}`);
    return path;
  }

  async getMacScreenResolution() {
    return new Promise((resolve, reject) => {
      exec("system_profiler SPDisplaysDataType | grep Resolution", (error, stdout, stderr) => {
        if (error) {
          reject(error);
          return;
        }
        // Example output:
        // Resolution: 1920 x 1080 (1080p FHD - Full High Definition)
        // Resolution: 3024 x 1964 Retina

        // We'll take the first resolution found for simplicity
        const lines = stdout.split("\n").filter(Boolean);
        if (lines.length === 0) {
          reject(new Error("No resolution info found"));
          return;
        }

        const match = lines[0].match(/Resolution:\s*(\d+)\s*x\s*(\d+)/i);
        if (!match) {
          reject(new Error("Could not parse resolution"));
          return;
        }

        const width = parseInt(match[1], 10);
        const height = parseInt(match[2], 10);

        resolve({ width, height });
      });
    });
  }

  async initialize(mode = "experimental", customConfig = {}) {
    try {
      // If viewport is not provided, try to dynamically get it for macOS
      if (!customConfig.viewport && process.platform === "darwin") {
        try {
          const resolution = await this.getMacScreenResolution();
          console.log(`🎯 Detected Mac screen resolution: ${resolution.width}x${resolution.height}`);
          customConfig.viewport = resolution;
        } catch (err) {
          console.warn("⚠️ Could not detect Mac screen resolution, falling back to default viewport", err.message);
        }
      }

      // Validate configuration
      const validatedConfig = StagehandConfigSchema.parse({
        mode,
        ...customConfig,
      });

      console.log(`🚀 Initializing Stagehand with Brave in ${validatedConfig.mode} mode...`);
      console.log(
        `📐 Viewport dimensions: ${validatedConfig.viewport?.width || this.viewport.width}x${validatedConfig.viewport?.height || this.viewport.height}`,
      );

      const bravePath = await this.getBravePath();

      const baseConfig = {
        env: "LOCAL",
        modelName: validatedConfig.modelName,
        modelClientOptions: {
          apiKey: process.env.MODEL_API_KEY || process.env.OPENAI_API_KEY,
        },
        verbose: validatedConfig.verbose,
        headless: validatedConfig.headless,
        selfHeal: validatedConfig.selfHeal,
        enableCaching: validatedConfig.enableCaching,
        waitForCaptchaSolves: validatedConfig.waitForCaptchaSolves,
        localBrowserLaunchOptions: {
          executablePath: bravePath,
          headless: false, // you already set this
          args: [
            "--no-first-run",
            "--disable-blink-features=AutomationControlled",
            "--disable-features=ChromeWhatsNewUI,ChromeLabs",
            "--start-maximized",
            "--private-access-tokens",
          ],
          ignoreDefaultArgs: ["--enable-automation"],
        },
      };

      // Configure mode-specific settings
      if (validatedConfig.mode === "experimental") {
        baseConfig.experimental = true;
        baseConfig.useAPI = false;
        console.log("📋 Using experimental mode (iframe support enabled)");
      } else {
        baseConfig.experimental = false;
        baseConfig.useAPI = true;
        console.log("🚀 Using API mode (optimized performance)");
      }

      this.stagehand = new Stagehand(baseConfig);

      console.log("🔄 Initializing Stagehand instance...");
      await this.stagehand.init();
      this.page = this.stagehand.page;
      this.sessionId = `session_${Date.now()}`;
      this.isInitialized = true;
      this.currentMode = validatedConfig.mode;

      console.log(`✅ Stagehand initialized in ${validatedConfig.mode} mode!`);
      console.log(`🆔 Session ID: ${this.sessionId}`);

      return {
        success: true,
        sessionId: this.sessionId,
        mode: validatedConfig.mode,
        viewport: validatedConfig.viewport || this.viewport,
      };
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      console.error("❌ Initialization failed:", message);

      if (error instanceof z.ZodError) {
        console.error("🔍 Validation errors:", error.errors);
        return {
          success: false,
          error: "Invalid configuration",
          validationErrors: error.errors,
        };
      }

      return { success: false, error: message };
    }
  }

  async switchMode(newMode) {
    try {
      // Validate mode
      const ModeSchema = z.enum(["api", "experimental"]);
      const validatedMode = ModeSchema.parse(newMode);

      if (this.currentMode === validatedMode) {
        console.log(`📋 Already in ${validatedMode} mode, no switch needed`);
        return { success: true, mode: validatedMode, switched: false };
      }

      console.log(`🔄 Switching from ${this.currentMode} to ${validatedMode} mode...`);
      console.log(`⏸️  Closing current session...`);

      // Close current session
      if (this.stagehand) {
        await this.stagehand.close();
      }

      // Reinitialize with new mode
      console.log(`🔄 Reinitializing with ${validatedMode} mode...`);
      const result = await this.initialize(validatedMode);

      if (result.success) {
        console.log(`✅ Successfully switched to ${validatedMode} mode`);
        return { success: true, mode: validatedMode, switched: true };
      } else {
        console.error(`❌ Failed to switch to ${validatedMode} mode:`, result.error);
        return { success: false, error: result.error };
      }
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      console.error(`❌ Error during mode switch:`, message);
      return { success: false, error: message };
    }
  }

  async getStatus() {
    if (!this.isInitialized) {
      return {
        initialized: false,
        message: "Session not initialized",
      };
    }

    try {
      const currentUrl = this.page.url();
      const title = await this.page.title();

      return {
        initialized: true,
        sessionId: this.sessionId,
        mode: this.currentMode,
        currentUrl,
        pageTitle: title,
        viewport: this.viewport,
      };
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      return {
        initialized: true,
        sessionId: this.sessionId,
        error: message,
      };
    }
  }

  async close() {
    try {
      console.log("🔚 Closing Stagehand session...");

      if (this.stagehand) {
        await this.stagehand.close();
        console.log("✅ Stagehand instance closed");
      }

      this.stagehand = null;
      this.page = null;
      this.sessionId = null;
      this.isInitialized = false;
      this.currentMode = null;

      console.log("✅ Session manager reset");
      return { success: true };
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      console.error("❌ Error during close:", message);
      return { success: false, error: message };
    }
  }

  getPage() {
    if (!this.isInitialized) {
      console.error("❌ Attempted to get page when session not initialized");
      throw new Error("Session not initialized");
    }
    return this.page;
  }

  getStagehand() {
    if (!this.isInitialized) {
      console.error("❌ Attempted to get Stagehand when session not initialized");
      throw new Error("Session not initialized");
    }
    return this.stagehand;
  }

  setupSignalHandlers() {
    process.on("SIGINT", () => {
      console.log("\n🛑 Received SIGINT (Ctrl+C)...");

      if (this.isInUserInput) {
        // If already in user input, shutdown
        console.log("💀 Shutting down from user input screen...");
        this.gracefulShutdown();
      } else {
        // Otherwise, trigger user input
        console.log("⏸️  Triggering user input...");
        this.triggerUserInput();
      }
    });

    process.on("SIGTERM", () => {
      console.log("\n🛑 Received SIGTERM...");
      this.gracefulShutdown();
    });
  }

  triggerUserInput() {
    this.isInUserInput = true;
    console.log("\n📝 =================");
    console.log("📝 USER INPUT MODE");
    console.log("📝 Type your message and press Enter to continue automation");
    console.log("📝 Press Ctrl+C again to shutdown");
    console.log("📝 =================");

    const readline = require("readline");
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    rl.question("💬 Your input: ", (input) => {
      this.isInUserInput = false;
      rl.close();

      if (input.trim()) {
        console.log(`✅ User provided: ${input}`);
        console.log("🔄 Resuming automation...");
      } else {
        console.log("📝 No input provided, resuming automation...");
      }
    });
  }

  async gracefulShutdown() {
    console.log("🛑 Initiating graceful shutdown...");

    try {
      if (this.stagehand) {
        console.log("🧹 Closing Stagehand session...");
        await this.stagehand.close();
      }
      console.log("✅ Shutdown complete");
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      console.error("❌ Error during shutdown:", message);
    }

    process.exit(0);
  }

  // New helper method inspired by LangChain's pattern
  async testBasicNavigation() {
    try {
      console.log("🧪 Running basic navigation test...");

      if (!this.isInitialized) {
        throw new Error("Session not initialized");
      }

      const testUrl = "https://www.example.com";
      console.log(`🧭 Navigating to test URL: ${testUrl}`);

      await this.page.goto(testUrl);
      const title = await this.page.title();
      const currentUrl = this.page.url();

      console.log(`✅ Navigation test successful`);
      console.log(`📄 Page title: ${title}`);
      console.log(`🔗 Current URL: ${currentUrl}`);

      return {
        success: true,
        url: currentUrl,
        title: title,
      };
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      console.error("❌ Test navigation failed:", message);
      return {
        success: false,
        error: message,
      };
    }
  }
}

module.exports = { StagehandSessionManager };
