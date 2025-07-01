// Simple tool schemas without validation
export const NavigateTool = {
  shape: {
    name: { value: "navigate" },
    description: { value: "Navigate to a URL" },
    arguments: { parse: (params: any) => params }
  }
};

// Simple JSON schemas for tools
export const navigateSchema = {
  type: "object",
  properties: {
    url: { type: "string", description: "URL to navigate to" }
  },
  required: ["url"]
};

export const goBackSchema = {
  type: "object",
  properties: {}
};

export const goForwardSchema = {
  type: "object", 
  properties: {}
};

export const pressKeySchema = {
  type: "object",
  properties: {
    key: { type: "string", description: "Key to press" }
  },
  required: ["key"]
};

export const waitSchema = {
  type: "object",
  properties: {
    time: { type: "number", description: "Time to wait in seconds" }
  },
  required: ["time"]
};

export const clickSchema = {
  type: "object",
  properties: {
    element: { type: "string", description: "Element to click" }
  },
  required: ["element"]
};

export const hoverSchema = {
  type: "object",
  properties: {
    element: { type: "string", description: "Element to hover over" }
  },
  required: ["element"]
};

export const typeSchema = {
  type: "object",
  properties: {
    element: { type: "string", description: "Element to type into" },
    text: { type: "string", description: "Text to type" }
  },
  required: ["element", "text"]
};

export const selectOptionSchema = {
  type: "object",
  properties: {
    element: { type: "string", description: "Select element" }
  },
  required: ["element"]
};

export const snapshotSchema = {
  type: "object",
  properties: {}
};

export const getConsoleLogsSchema = {
  type: "object",
  properties: {}
};

export const screenshotSchema = {
  type: "object",
  properties: {}
};

export const dragSchema = {
  type: "object",
  properties: {
    startElement: { type: "string", description: "Element to drag from" },
    endElement: { type: "string", description: "Element to drag to" }
  },
  required: ["startElement", "endElement"]
};

export const GoBackTool = {
  shape: {
    name: { value: "go_back" },
    description: { value: "Go back in browser history" },
    arguments: { parse: (params: any) => params }
  }
};

export const GoForwardTool = {
  shape: {
    name: { value: "go_forward" },
    description: { value: "Go forward in browser history" },
    arguments: { parse: (params: any) => params }
  }
};

export const PressKeyTool = {
  shape: {
    name: { value: "press_key" },
    description: { value: "Press a key" },
    arguments: { parse: (params: any) => params }
  }
};

export const WaitTool = {
  shape: {
    name: { value: "wait" },
    description: { value: "Wait for specified time" },
    arguments: { parse: (params: any) => params }
  }
};

export const ClickTool = {
  shape: {
    name: { value: "click" },
    description: { value: "Click an element" },
    arguments: { parse: (params: any) => params }
  }
};

export const HoverTool = {
  shape: {
    name: { value: "hover" },
    description: { value: "Hover over an element" },
    arguments: { parse: (params: any) => params }
  }
};

export const TypeTool = {
  shape: {
    name: { value: "type" },
    description: { value: "Type text into an element" },
    arguments: { parse: (params: any) => params }
  }
};

export const SelectOptionTool = {
  shape: {
    name: { value: "select_option" },
    description: { value: "Select an option from a dropdown" },
    arguments: { parse: (params: any) => params }
  }
};

export const SnapshotTool = {
  shape: {
    name: { value: "snapshot" },
    description: { value: "Take a snapshot of the page" },
    arguments: { parse: (params: any) => params }
  }
};

export const GetConsoleLogsTool = {
  shape: {
    name: { value: "get_console_logs" },
    description: { value: "Get browser console logs" },
    arguments: { parse: (params: any) => params }
  }
};

export const ScreenshotTool = {
  shape: {
    name: { value: "screenshot" },
    description: { value: "Take a screenshot" },
    arguments: { parse: (params: any) => params }
  }
};

export const DragTool = {
  shape: {
    name: { value: "drag" },
    description: { value: "Drag an element" },
    arguments: { parse: (params: any) => params }
  }
};