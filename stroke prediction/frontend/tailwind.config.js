/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#2CA58D",
        secondary: "#0F766E",
        dark: "#0F172A",
        light: "#F8FBFB",
        textPrimary: "#0F172A",
        textSecondary: "#475569"
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
