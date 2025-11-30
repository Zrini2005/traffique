/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#174D38',
        secondary: '#2C7A5F',
      },
    },
  },
  plugins: [],
}
