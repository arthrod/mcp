// bridge-server.js - Express server that uses the session manager
console.log("🔍 Debug: Starting bridge-server imports");
const express = require("express");
const { z } = require("zod");
console.log("🔍 Debug: About to import StagehandSessionManager");
const { StagehandSessionManager } = require("./session-manager.js");
console.log("🔍 Debug: StagehandSessionManager imported successfully");

const app = express();
app.use(express.json());

// Global session manager instance
let sessionManager = null;

// Zod schemas for request validation (inspired by LangChain patterns)
const NavigateRequestSchema = z.object({
  url: z.string().url().describe("Valid URL to navigate to"),
});

const ActRequestSchema = z.object({
  action: z.string().min(1).describe("Action to perform on the page"),
});

const ObserveRequestSchema = z.object({
  instruction: z.string().min(1).describe("Instruction for observation"),
});

const ExtractRequestSchema = z.object({
  instruction: z.string().min(1).describe("Extraction instruction"),
  schema: z.record(z.any()).optional().describe("Extraction schema"),
});

const ExtractDomRequestSchema = z.object({
  instruction: z.string().optional().describe("Extraction instruction"),
  selectors: z.array(z.string()).optional().describe("CSS selectors to try"),
  useIframes: z.boolean().optional().describe("Whether to search in iframes"),
});

const PasteClipboardRequestSchema = z.object({
  content: z.string().min(1).describe("Content to paste"),
  target_description: z.string().default("document content area").describe("Target area description"),
});

const AgentRequestSchema = z.object({
  instructions: z.string().min(1).describe("Agent instructions"),
  model: z.string().default("computer-use-preview").describe("Model to use"),
  maxSteps: z.number().min(1).max(50).default(5).describe("Maximum steps"),
});

const VerifyVisualRequestSchema = z.object({
  description: z.string().min(1).describe("What to verify visually"),
});

// Type guard for error handling (from LangChain pattern)
function isErrorWithMessage(error) {
  return typeof error === "object" && error !== null && "message" in error && typeof error.message === "string";
}

// Helper function to handle Zod validation errors
function handleValidationError(error, res) {
  if (error instanceof z.ZodError) {
    console.error("❌ Validation error:", error.errors);
    return res.status(400).json({
      success: false,
      error: "Validation failed",
      details: error.errors,
    });
  }
  return null;
}

// Middleware to check if session is initialized
const requireSession = (req, res, next) => {
  if (!sessionManager || !sessionManager.isInitialized) {
    console.warn("⚠️  Request made without initialized session");
    return res.status(400).json({
      success: false,
      error: "Session not initialized. Call /init first.",
    });
  }
  next();
};

// Middleware for request logging
const logRequest = (req, res, next) => {
  console.log(`📥 ${req.method} ${req.path} - ${new Date().toISOString()}`);
  if (Object.keys(req.body).length > 0) {
    console.log(`📋 Request body:`, JSON.stringify(req.body, null, 2));
  }
  next();
};

app.use(logRequest);

// INTEGRATION INSTRUCTIONS:
// 1. Add the InjectPromptRequestSchema to the validation schemas section at the top of bridge-server.js
// 2. Copy the two endpoint definitions below and paste them before the error handling middleware
// 3. Update the 404 handler's availableRoutes array to include the new routes
// 4. The endpoints will automatically use the existing sessionManager and error handling

// Add this to the validation schemas section:

// Add these endpoints to bridge-server.js after the existing endpoints

// Inject prompt directly into input field
app.post("/inject-prompt", requireSession, async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: "No prompt provided",
      });
    }

    console.log(`💉 Injecting prompt: ${prompt.substring(0, 50)}...`);
    console.log(`📏 Prompt length: ${prompt.length} characters`);

    const page = sessionManager.getPage();

    // Method 1: Direct value injection with React event triggering
    try {
      console.log("🔧 Method 1: Direct injection with React events");

      const injected = await page.evaluate(async (promptText) => {
        // Find the input field - ChatGPT uses contenteditable
        const selectors = [
          "#prompt-textarea", // ChatGPT's main textarea
          'textarea[placeholder*="Message"]',
          'div[contenteditable="true"]',
          'div[role="textbox"]',
          ".ProseMirror", // Common in modern editors
          "p[data-placeholder]", // Placeholder paragraphs
        ];

        let inputElement = null;
        let method = "";

        // Try each selector
        for (const selector of selectors) {
          const element = document.querySelector(selector);
          if (element) {
            inputElement = element;
            method = selector;
            break;
          }
        }

        if (!inputElement) {
          // Fallback: find by traversing from composer button
          const composer = document.querySelector('[aria-label="Message Composer"]');
          if (composer) {
            inputElement = composer.querySelector('p, div[contenteditable="true"], textarea');
            method = "composer-child";
          }
        }

        if (!inputElement) {
          return { success: false, error: "No input field found", tried: selectors };
        }

        // Clear existing content
        if (inputElement.tagName === "TEXTAREA" || inputElement.tagName === "INPUT") {
          // Regular input/textarea
          inputElement.focus();
          inputElement.value = "";
          inputElement.value = promptText;

          // Trigger React events
          const inputEvent = new Event("input", { bubbles: true, cancelable: true });
          const changeEvent = new Event("change", { bubbles: true, cancelable: true });
          inputElement.dispatchEvent(inputEvent);
          inputElement.dispatchEvent(changeEvent);
        } else {
          // ContentEditable div/p
          inputElement.focus();
          inputElement.textContent = promptText;
          inputElement.innerText = promptText;

          // Trigger input event for contenteditable
          const inputEvent = new InputEvent("input", {
            bubbles: true,
            cancelable: true,
            inputType: "insertText",
            data: promptText,
          });
          inputElement.dispatchEvent(inputEvent);

          // Also trigger these for good measure
          inputElement.dispatchEvent(new Event("change", { bubbles: true }));
          inputElement.dispatchEvent(new Event("keyup", { bubbles: true }));
        }

        return {
          success: true,
          method: method,
          elementType: inputElement.tagName,
          contentLength: promptText.length,
        };
      }, prompt);

      if (injected.success) {
        console.log(`✅ Injection successful via ${injected.method} (${injected.elementType})`);
        console.log(`📝 Injected ${injected.contentLength} characters`);

        // Small delay to ensure React processes the change
        await page.waitForTimeout(500);

        res.json({
          success: true,
          message: "Prompt injected successfully",
          method: injected.method,
          elementType: injected.elementType,
        });
        return;
      } else {
        console.log(`⚠️ Method 1 failed: ${injected.error}`);
        console.log(`🔍 Tried selectors: ${JSON.stringify(injected.tried)}`);
      }
    } catch (error) {
      console.log(`⚠️ Method 1 error: ${error.message}`);
    }

    // Method 2: Click and paste via clipboard (fallback)
    console.log("🔧 Method 2: Clipboard injection fallback");

    // Set clipboard content
    await page.evaluate(async (text) => {
      await navigator.clipboard.writeText(text);
    }, prompt);
    console.log("📋 Prompt copied to clipboard");

    // Click on the input area
    await page.click('[aria-label="Message Composer"]').catch(() => {});
    await page.waitForTimeout(300);

    // Paste using keyboard shortcut
    const isMac = process.platform === "darwin";
    await page.keyboard.press(isMac ? "Meta+V" : "Control+V");
    console.log(`✅ Pasted via ${isMac ? "Cmd" : "Ctrl"}+V`);

    res.json({
      success: true,
      message: "Prompt injected via clipboard",
      method: "clipboard",
    });

    /* Method 3: Form value setting (commented out as requested)
    console.log("🔧 Method 3: Form value setting");

    // Find the form and set value directly
    const formSet = await page.evaluate((promptText) => {
      const form = document.querySelector('form');
      if (form) {
        const input = form.querySelector('textarea, input[type="text"], div[contenteditable="true"]');
        if (input) {
          if (input.tagName === 'TEXTAREA' || input.tagName === 'INPUT') {
            input.value = promptText;
          } else {
            input.textContent = promptText;
          }

          // Submit form
          const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
          form.dispatchEvent(submitEvent);
          return true;
        }
      }
      return false;
    }, prompt);

    if (formSet) {
      console.log("✅ Form value set and submitted");
    }
    */
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Injection error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Extract AI response using direct DOM scraping
app.post("/extract-response", requireSession, async (req, res) => {
  try {
    console.log("🎯 Extracting AI response from DOM");

    const page = sessionManager.getPage();
    let extractedResponse = null;

    // Method 1: Direct DOM scraping
    console.log("🔧 Method 1: Direct DOM scraping");

    const domContent = await page.evaluate(() => {
      // Selectors for ChatGPT responses
      const responseSelectors = [
        // ChatGPT specific
        'div[data-message-author-role="assistant"]',
        "div.agent-turn",
        "div.markdown.prose",
        ".text-message:not(.user-message)",
        'div[class*="assistant"]',

        // Generic selectors
        "div.response-content",
        "div.ai-response",
        "div.bot-message",
        '[data-testid="bot-message"]',
      ];

      // Try each selector
      for (const selector of responseSelectors) {
        const elements = document.querySelectorAll(selector);
        if (elements.length > 0) {
          // Get the last response (most recent)
          const lastElement = elements[elements.length - 1];
          const text = lastElement.innerText || lastElement.textContent || "";

          if (text.trim().length > 10) {
            return {
              success: true,
              response: text.trim(),
              selector: selector,
              elementCount: elements.length,
              method: "dom_scraping",
            };
          }
        }
      }

      // Fallback: Look for any element containing substantial text after user message
      const allElements = document.querySelectorAll("p, div, span");
      const texts = [];

      for (const el of allElements) {
        const text = el.innerText || el.textContent || "";
        if (text.length > 50 && !text.includes("Message") && !text.includes("ChatGPT")) {
          texts.push({
            text: text.trim(),
            tag: el.tagName,
            className: el.className,
          });
        }
      }

      if (texts.length > 0) {
        // Return the longest text as likely response
        const longest = texts.reduce((a, b) => (a.text.length > b.text.length ? a : b));
        return {
          success: true,
          response: longest.text,
          selector: `${longest.tag}.${longest.className}`,
          method: "text_search",
        };
      }

      return { success: false, error: "No response content found" };
    });

    if (domContent.success) {
      console.log(`✅ DOM extraction successful via ${domContent.method}`);
      console.log(`📍 Used selector: ${domContent.selector}`);
      console.log(`📏 Response length: ${domContent.response.length} characters`);
      console.log(`📄 Preview: ${domContent.response.substring(0, 100)}...`);

      extractedResponse = domContent;
    } else {
      console.log("⚠️ Method 1 failed: No response found via DOM");
    }

    // Method 2: Clipboard extraction (fallback)
    if (!extractedResponse) {
      console.log("🔧 Method 2: Clipboard extraction fallback");

      try {
        // Select all text in the response area
        await page.keyboard.press("Control+a");
        await page.waitForTimeout(300);

        // Copy to clipboard
        const isMac = process.platform === "darwin";
        await page.keyboard.press(isMac ? "Meta+c" : "Control+c");
        await page.waitForTimeout(300);

        // Read from clipboard
        const clipboardContent = await page.evaluate(async () => {
          try {
            const text = await navigator.clipboard.readText();
            return text;
          } catch (e) {
            return null;
          }
        });

        if (clipboardContent && clipboardContent.length > 50) {
          console.log(`✅ Clipboard extraction successful`);
          console.log(`📏 Response length: ${clipboardContent.length} characters`);

          extractedResponse = {
            success: true,
            response: clipboardContent,
            method: "clipboard",
          };
        } else {
          console.log("⚠️ Method 2 failed: Clipboard empty or too short");
        }
      } catch (clipError) {
        console.log(`⚠️ Clipboard extraction error: ${clipError.message}`);
      }
    }

    // Return result
    if (extractedResponse) {
      res.json({
        success: true,
        data: extractedResponse,
        message: `Response extracted via ${extractedResponse.method}`,
      });
    } else {
      console.log("❌ All extraction methods failed");
      res.json({
        success: false,
        error: "Could not extract response",
        message: "Try waiting longer or checking if response is visible",
      });
    }
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Response extraction error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Update the 404 handler's availableRoutes array to include:
// "POST /inject-prompt",
// "POST /extract-response",
// Health check endpoint
app.get("/health", (req, res) => {
  const healthStatus = {
    success: true,
    message: "Bridge server is running",
    timestamp: new Date().toISOString(),
    sessionActive: sessionManager ? sessionManager.isInitialized : false,
    version: "2.0.0", // Version with Zod validation
  };

  console.log("💚 Health check:", healthStatus);
  res.json(healthStatus);
});

// Initialize session
app.post("/init", async (req, res) => {
  console.log("🔍 Debug: Entered /init endpoint");
  try {
    console.log("🔄 Received init request");

    // Close existing session if any
    if (sessionManager && sessionManager.isInitialized) {
      console.log("🧹 Closing existing initialized session...");
      try {
        await sessionManager.close();
        console.log("🔍 Debug: Existing session closed successfully");
      } catch (closeError) {
        console.error("🔍 Debug: Error closing existing session:", closeError);
      }
    }

    // Reset sessionManager to null to ensure clean state
    sessionManager = null;

    console.log("🔍 Debug: Creating new StagehandSessionManager");
    // Create new session manager
    try {
      sessionManager = new StagehandSessionManager();
      console.log("🔍 Debug: StagehandSessionManager constructor completed successfully");
    } catch (constructorError) {
      console.error("🔍 Debug: StagehandSessionManager constructor failed:", constructorError);
      throw constructorError;
    }
    
    console.log("🔍 Debug: StagehandSessionManager created, calling initialize()");
    const result = await sessionManager.initialize();
    console.log("🔍 Debug: Initialize completed, result:", result);

    console.log("📤 Sending init response:", result.success ? "✅" : "❌");
    res.json(result);
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Init error:", message);
    console.error("🔍 Debug: Full error object:", error);
    console.error("🔍 Debug: Error stack:", error.stack);
    res.status(500).json({
      success: false,
      error: message,
      stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
    });
  }
});

// Get session status
app.get("/status", async (req, res) => {
  try {
    if (!sessionManager) {
      console.log("ℹ️  No session manager exists");
      return res.json({
        success: true,
        status: "No session created",
      });
    }

    const status = await sessionManager.getStatus();
    console.log("📊 Session status:", status);
    res.json({
      success: true,
      status: status,
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Status error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Navigate to URL
app.post("/navigate", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = NavigateRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { url } = validation.data;
    console.log(`🧭 Navigating to: ${url}`);

    const page = sessionManager.getPage();
    const startTime = Date.now();
    await page.goto(url);
    const loadTime = Date.now() - startTime;

    const result = {
      success: true,
      message: `Navigated to ${url}`,
      url: page.url(),
      title: await page.title(),
      loadTimeMs: loadTime,
    };

    console.log(`✅ Navigation successful in ${loadTime}ms`);
    res.json(result);
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Navigation error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Perform action
app.post("/act", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = ActRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { action } = validation.data;
    console.log(`🎬 Performing action: ${action}`);

    const page = sessionManager.getPage();
    const startTime = Date.now();
    const result = await page.act(action);
    const executionTime = Date.now() - startTime;

    console.log(`✅ Action completed in ${executionTime}ms`);
    res.json({
      success: true,
      data: result,
      message: `Action completed: ${action}`,
      executionTimeMs: executionTime,
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Action error:", message);
    res.status(500).json({
      success: false,
      error: message,
      action: req.body.action,
    });
  }
});

// Observe page
app.post("/observe", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = ObserveRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { instruction } = validation.data;
    console.log(`👁️  Observing: ${instruction}`);

    const page = sessionManager.getPage();
    const observeInstruction = `Look at the current page and check if this content is visible: ${instruction}. Focus on clickable items, text content, document body, and any readable text.`;

    console.log(`🔍 Full observation instruction: ${observeInstruction}`);
    const result = await page.observe(observeInstruction, { screenshot: true });

    console.log(`✅ Observation completed, found ${Array.isArray(result) ? result.length : 0} items`);
    res.json({
      success: true,
      data: result,
      message: `Observation completed: ${instruction}`,
      itemCount: Array.isArray(result) ? result.length : 0,
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Observation error:", message);
    res.status(500).json({
      success: false,
      error: message,
      instruction: req.body.instruction,
    });
  }
});

// Extract data
app.post("/extract", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = ExtractRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { instruction, schema } = validation.data;
    console.log(`📊 Extracting: ${instruction}`);
    if (schema) {
      console.log(`📋 Using schema:`, JSON.stringify(schema, null, 2));
    }

    const page = sessionManager.getPage();
    const result = await page.extract({
      instruction,
      schema: schema,
    });

    console.log("✅ Extraction completed");
    res.json({
      success: true,
      data: result,
      message: `Extraction completed: ${instruction}`,
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Extraction error:", message);
    res.status(500).json({
      success: false,
      error: message,
      instruction: req.body.instruction,
    });
  }
});

// Clipboard-based paste endpoint
app.post("/paste-clipboard", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = PasteClipboardRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { content, target_description } = validation.data;
    console.log(`📋 Clipboard paste: ${content.substring(0, 100)}... to ${target_description}`);
    console.log(`📏 Content length: ${content.length} characters`);

    const page = sessionManager.getPage();

    // Step 1: Grant clipboard permissions
    const context = page.context();
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    console.log("✅ Clipboard permissions granted");

    // Step 2: Set clipboard content programmatically (with user activation)
    try {
      // Create user activation by simulating click
      await page.evaluate(() => {
        document.body.click();
      });

      await page.evaluate(async (text) => {
        await navigator.clipboard.writeText(text);
      }, content);
      console.log("✅ Content written to clipboard");
    } catch (clipboardError) {
      console.log("⚠️ Direct clipboard write failed, using alternative method");
      // Fallback: Use document focus and retry
      await page.focus("body");
      await page.waitForTimeout(500);

      try {
        await page.evaluate(async (text) => {
          await navigator.clipboard.writeText(text);
        }, content);
        console.log("✅ Content written to clipboard (fallback method)");
      } catch (e) {
        const errorMsg = isErrorWithMessage(e) ? e.message : String(e);
        console.log("❌ Both clipboard methods failed:", errorMsg);
        throw new Error(`Clipboard write failed: ${errorMsg}`);
      }
    }

    // Step 3: Focus the target area (for Google Docs, multiple attempts)
    if (target_description.includes("document") || target_description.includes("Google")) {
      console.log("📄 Detected document target, trying multiple focus strategies...");

      // Try multiple selectors for Google Docs focus
      const selectors = [
        'canvas[class*="kix-canvas"]',
        '[role="textbox"]',
        '[contenteditable="true"]',
        ".kix-appview-editor",
        "canvas",
        "body",
      ];

      let focused = false;
      for (const selector of selectors) {
        try {
          await page.click(selector, { timeout: 2000 });
          await page.waitForTimeout(500);
          console.log(`✅ Focused using selector: ${selector}`);
          focused = true;
          break;
        } catch (e) {
          console.log(`⚠️ Focus attempt failed for ${selector}`);
        }
      }

      if (!focused) {
        // Force focus using JavaScript
        await page.evaluate(() => {
          document.body.focus();
          document.body.click();
        });
        console.log("✅ Force focused using JavaScript");
      }

      // Additional Mac-specific focus handling
      const isMac = process.platform === "darwin";
      if (isMac) {
        await page.keyboard.press("Tab");
        await page.waitForTimeout(300);
        console.log("✅ Mac: Additional Tab focus");
      }
    }

    // Step 4: Use keyboard shortcut to paste (cross-platform)
    const isMac = process.platform === "darwin";
    const modifier = isMac ? "Meta" : "Control";

    // For Mac, try multiple paste approaches
    if (isMac) {
      console.log("🍎 Mac detected - using enhanced paste sequence");
      await page.keyboard.press("Meta+V");
      await page.waitForTimeout(500);
      // Fallback: try Command+Shift+V (paste without formatting)
      await page.keyboard.press("Meta+Shift+V");
      console.log(`✅ Mac paste executed (Meta+V + Meta+Shift+V)`);
    } else {
      await page.keyboard.press("Control+V");
      console.log(`✅ Windows/Linux paste executed (Control+V)`);
    }

    // Step 5: Wait for paste to complete
    await page.waitForTimeout(1500);

    console.log("✅ Clipboard paste completed");
    res.json({
      success: true,
      message: "Content pasted via clipboard",
      method: "keyboard_shortcut",
      modifier_used: modifier,
      contentLength: content.length,
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Clipboard paste error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Agent execution (for complex multi-step tasks)
app.post("/agent", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = AgentRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { instructions, model, maxSteps } = validation.data;
    console.log(`🤖 Agent executing: ${instructions.substring(0, 100)}...`);
    console.log(`🔧 Configuration: model=${model}, maxSteps=${maxSteps}`);

    const stagehand = sessionManager.getStagehand();
    const agent = stagehand.agent({
      model: model,
      instructions:
        "You are a helpful web navigation assistant that helps users find information. Do not ask follow up questions, the user will trust your judgement.",
      maxSteps: maxSteps,
      waitBetweenActions: 100,
      context:
        "CRITICAL NOTEBOOKLM INSTRUCTIONS: To add content: 1) Go to 'Add sources', 2) Look for 'Paste text' section, 3) Below 'Paste text' click 'Copied text' button, 4) Type/paste content, 5) Submit/save. For chat: 1) Type question in chat field, 2) Click the arrow/send button (→), 3) Wait for response to generate, 4) Extract when complete. CRITICAL GOOGLE DOCS: When in new document there are TWO forms: 1) FIRST form is for title, 2) SECOND form is for content. Paste into SECOND form (content area), not title. User is on Mac, use cmd not ctrl.",
      options: {
        apiKey: process.env.OPENAI_API_KEY,
      },
    });

    const startTime = Date.now();
    const result = await agent.execute(instructions);
    const executionTime = Date.now() - startTime;

    console.log(`✅ Agent execution completed in ${executionTime}ms`);
    res.json({
      success: true,
      data: result,
      message: "Agent execution completed",
      executionTimeMs: executionTime,
      stepsUsed: result.steps?.length || 0,
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Agent error:", message);
    res.status(500).json({
      success: false,
      error: message,
      instructions: req.body.instructions,
    });
  }
});

// Enhanced extraction endpoint using proper Stagehand extract() with iframe support
app.post("/extract-dom", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = ExtractDomRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { instruction, selectors, useIframes } = validation.data;
    const page = sessionManager.getPage();
    const stagehand = sessionManager.getStagehand();

    console.log(`📊 Enhanced extraction starting`);
    console.log(`🔍 Current page URL: ${page.url()}`);
    console.log(`⏰ Starting extraction at: ${new Date().toISOString()}`);

    // Auto-detect iframes if not explicitly specified
    let shouldUseIframes = useIframes;
    if (shouldUseIframes === undefined) {
      try {
        const iframeCount = await page.evaluate(() => {
          return document.querySelectorAll("iframe").length;
        });
        shouldUseIframes = iframeCount > 0;
        console.log(`🔍 Auto-detected ${iframeCount} iframes, setting iframes: ${shouldUseIframes}`);
      } catch (e) {
        console.log(`⚠️ Iframe detection failed: ${e.message}`);
        shouldUseIframes = false;
      }
    }

    let result = null;

    // Method 1: Try Stagehand's intelligent extract first
    if (instruction) {
      try {
        console.log(`🤖 Attempting Stagehand extract with instruction: "${instruction}"`);

        const extractOptions = {
          instruction: instruction,
          schema: require("zod").object({
            content: require("zod").string().describe("The extracted text content"),
            source: require("zod").string().optional().describe("Where the content was found"),
          }),
          iframes: shouldUseIframes,
          domSettleTimeoutMs: 2000,
        };

        console.log(`📋 Extract options:`, { ...extractOptions, schema: "[zod schema]" });

        const stagehandResult = await stagehand.page.extract(extractOptions);

        if (stagehandResult && stagehandResult.content && stagehandResult.content.trim().length > 0) {
          result = {
            method: "stagehand_extract",
            content: stagehandResult.content,
            source: stagehandResult.source || "stagehand_extraction",
            combinedText: stagehandResult.content,
            extractedLength: stagehandResult.content.length,
            usedIframes: shouldUseIframes,
          };
          console.log(`✅ Stagehand extract succeeded: ${result.extractedLength} characters`);
        } else {
          console.log(`⚠️ Stagehand extract returned empty content`);
        }
      } catch (stagehandError) {
        const errorMsg = isErrorWithMessage(stagehandError) ? stagehandError.message : String(stagehandError);
        console.log(`⚠️ Stagehand extract failed: ${errorMsg}`);
      }
    }

    // Method 2: Fallback to selector-based extraction if Stagehand failed
    if (!result && selectors) {
      console.log(`🔄 Falling back to selector-based extraction`);

      const selectorsToTry = Array.isArray(selectors)
        ? selectors
        : [".response-text", '[data-testid="response"]', "p", 'div[class*="response"]', 'span[class*="text"]'];

      console.log(`🎯 Will attempt ${selectorsToTry.length} selectors`);

      const selectorResult = await page.evaluate((selectorList) => {
        for (const selector of selectorList) {
          try {
            let elements;

            if (selector.startsWith("xpath=")) {
              const xpath = selector.replace("xpath=", "");
              const xpathResult = document.evaluate(xpath, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
              elements = [];
              for (let i = 0; i < xpathResult.snapshotLength; i++) {
                elements.push(xpathResult.snapshotItem(i));
              }
            } else {
              elements = Array.from(document.querySelectorAll(selector));
            }

            if (elements && elements.length > 0) {
              const texts = elements
                .map((el) => {
                  const text = el.innerText || el.textContent || "";
                  return text.trim();
                })
                .filter((text) => text.length > 0);

              if (texts.length > 0) {
                return {
                  selector: selector,
                  texts: texts,
                  combinedText: texts.join("\n\n"),
                  elementCount: elements.length,
                  textCount: texts.length,
                };
              }
            }
          } catch (e) {
            console.log(`❌ Selector ${selector} failed: ${e.message}`);
          }
        }
        return null;
      }, selectorsToTry);

      if (selectorResult) {
        result = {
          method: "selector_extraction",
          ...selectorResult,
          extractedLength: selectorResult.combinedText?.length || 0,
          usedIframes: shouldUseIframes,
        };
        console.log(`✅ Selector extraction succeeded: ${result.extractedLength} characters`);
        console.log(`📍 Used selector: ${selectorResult.selector}`);
      }
    }

    if (result) {
      console.log(`✅ Extraction completed successfully via ${result.method}`);
      console.log(`📄 Extracted text length: ${result.extractedLength} characters`);
      console.log(`📊 Preview: ${result.combinedText?.substring(0, 150)}...`);
    } else {
      console.log(`⚠️ All extraction methods failed`);
    }

    res.json({
      success: true,
      data: result,
      message: result ? `Extraction completed via ${result.method}` : "No content found with any method",
      extractedLength: result?.extractedLength || 0,
      usedIframes: shouldUseIframes,
      method: result?.method || "none",
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Enhanced extraction error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Visual verification endpoint for canvas-based content (like Google Docs)
app.post("/verify-visual", requireSession, async (req, res) => {
  try {
    // Validate request
    const validation = VerifyVisualRequestSchema.safeParse(req.body);
    if (!validation.success) {
      return handleValidationError(validation.error, res);
    }

    const { description } = validation.data;
    console.log(`📸 Visual verification: ${description}`);

    const page = sessionManager.getPage();
    const stagehand = sessionManager.getStagehand();

    // Take screenshot first
    console.log("📷 Taking screenshot...");
    const screenshot = await page.screenshot({
      type: "png",
      encoding: "base64",
      fullPage: false,
    });
    console.log("📷 Screenshot captured");

    // Use Stagehand's observe with screenshot for better analysis
    console.log("🔍 Running Stagehand observe...");
    const observeResult = await stagehand.page.observe(
      `Look at the current page and check if this content is visible: ${description}. Focus on text content, document body, and any readable text.`,
      { screenshot: true },
    );
    console.log("🔍 Stagehand observe result:", observeResult);

    // Also try direct text-based verification
    let textFound = false;
    try {
      console.log("📄 Attempting text extraction...");
      const pageText = await page.evaluate(() => {
        // Try multiple ways to get page text
        const bodyText = document.body.innerText || document.body.textContent || "";
        const canvasTexts = Array.from(document.querySelectorAll("canvas"))
          .map((c) => c.getAttribute("aria-label") || "")
          .join(" ");
        const editableTexts = Array.from(document.querySelectorAll('[contenteditable="true"]'))
          .map((e) => e.innerText || e.textContent || "")
          .join(" ");

        return {
          bodyText: bodyText.substring(0, 1000),
          canvasTexts: canvasTexts.substring(0, 500),
          editableTexts: editableTexts.substring(0, 500),
          totalLength: bodyText.length,
        };
      });

      console.log("📄 Page text analysis:");
      console.log(`  - Body text length: ${pageText.totalLength}`);
      console.log(`  - Canvas text found: ${pageText.canvasTexts.length > 0 ? "Yes" : "No"}`);
      console.log(`  - Editable text found: ${pageText.editableTexts.length > 0 ? "Yes" : "No"}`);

      // Check if any meaningful text content is found
      const allText = (pageText.bodyText + " " + pageText.canvasTexts + " " + pageText.editableTexts).trim();
      textFound = allText.length > 50; // Check if substantial text content exists

      console.log(`🔍 Text search result: ${textFound ? "FOUND" : "NOT_FOUND"} substantial text content (${allText.length} chars)`);
    } catch (textError) {
      const errorMsg = isErrorWithMessage(textError) ? textError.message : String(textError);
      console.log("⚠️ Text extraction failed:", errorMsg);
    }

    // Determine verification result
    const verified = textFound || (observeResult && observeResult.length > 0);

    console.log(`✅ Visual verification completed: ${verified ? "VERIFIED" : "NOT_FOUND"}`);
    res.json({
      success: true,
      data: {
        screenshot: screenshot,
        analysis: observeResult,
        textAnalysis: textFound ? "Key terms found in document" : "Key terms not found",
        verified: verified,
      },
      message: verified ? "Content verified as visible" : "Content not found or not visible",
    });
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Visual verification error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Test navigation endpoint
app.post("/test", requireSession, async (req, res) => {
  try {
    console.log("🧪 Running test navigation...");
    const result = await sessionManager.testBasicNavigation();
    res.json(result);
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Test error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Close session
app.post("/close", async (req, res) => {
  try {
    console.log("🔚 Closing session...");

    let result = { success: true, message: "No session to close" };

    if (sessionManager) {
      result = await sessionManager.close();
      sessionManager = null;
    }

    console.log("✅ Session closed");
    res.json(result);
  } catch (error) {
    const message = isErrorWithMessage(error) ? error.message : String(error);
    console.error("❌ Close error:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  const message = isErrorWithMessage(err) ? err.message : String(err);
  console.error("💥 Unhandled error:", message);
  res.status(500).json({
    success: false,
    error: message,
    stack: process.env.NODE_ENV === "development" ? err.stack : undefined,
  });
});

// 404 handler
app.use((req, res) => {
  console.warn(`⚠️  404 - Route not found: ${req.method} ${req.path}`);
  res.status(404).json({
    success: false,
    error: "Route not found",
    availableRoutes: [
      "POST /init",
      "GET /status",
      "POST /navigate",
      "POST /act",
      "POST /observe",
      "POST /extract",
      "POST /extract-dom",
      "POST /paste-clipboard",
      "POST /agent",
      "POST /verify-visual",
      "POST /test",
      "POST /close",
      "GET /health",
    ],
  });
});

// Graceful shutdown
process.on("SIGINT", async () => {
  console.log("\n🛑 Received SIGINT, shutting down gracefully...");

  if (sessionManager) {
    try {
      await sessionManager.close();
      console.log("✅ Session closed during shutdown");
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      console.error("❌ Error during shutdown:", message);
    }
  }

  process.exit(0);
});

process.on("SIGTERM", async () => {
  console.log("\n🛑 Received SIGTERM, shutting down gracefully...");

  if (sessionManager) {
    try {
      await sessionManager.close();
      console.log("✅ Session closed during shutdown");
    } catch (error) {
      const message = isErrorWithMessage(error) ? error.message : String(error);
      console.error("❌ Error during shutdown:", message);
    }
  }

  process.exit(0);
});

// Start server
const PORT = process.env.PORT || 3001;
const server = app.listen(PORT, () => {
  console.log(`🚀 Stagehand Bridge Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`🔧 Initialize session: POST http://localhost:${PORT}/init`);
  console.log(`📈 Check status: GET http://localhost:${PORT}/status`);
  console.log("");
  console.log("💡 Available endpoints:");
  console.log("  POST /init        - Initialize Stagehand session");
  console.log("  GET  /status      - Get session status");
  console.log("  POST /navigate    - Navigate to URL");
  console.log("  POST /act         - Perform action");
  console.log("  POST /observe     - Observe page");
  console.log("  POST /extract     - Extract data");
  console.log("  POST /extract-dom - Enhanced extraction with iframe support");
  console.log("  POST /paste-clipboard - Paste content via clipboard");
  console.log("  POST /agent       - Execute agent task");
  console.log("  POST /verify-visual - Visual verification with screenshot");
  console.log("  POST /test        - Test navigation");
  console.log("  POST /close       - Close session");
  console.log("  GET  /health      - Health check");
  console.log("");
  console.log("🔒 Input validation: All endpoints now use Zod schemas");
  console.log("📐 Viewport: 1600x1067 (XDR Display P3)");
  console.log("🤖 Model: gpt-4.1");
});

module.exports = { app, server };
