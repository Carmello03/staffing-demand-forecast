import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          700: "#1f2937",
          900: "#0f172a",
        },
        brand: {
          50: "#eef9ff",
          100: "#d7f1ff",
          300: "#7fcdf2",
          500: "#2584c9",
          700: "#1b5d9d",
          900: "#103d6f",
        },
        accent: {
          50: "#fff6ed",
          100: "#ffe9d6",
          500: "#eb8c44",
          700: "#b9571f",
        },
      },
      boxShadow: {
        glow: "0 24px 55px -20px rgba(37, 132, 201, 0.35)",
      },
    },
  },
  plugins: [],
} satisfies Config;
