# Staffing Demand Forecast Frontend

React + TypeScript single-page dashboard using Vite, Tailwind CSS, React Router, and Firebase Auth.

## 1) Setup

```bash
cd frontend
npm install
```

Create `.env` from `.env.example` and fill values:

- `VITE_API_BASE_URL`: backend base URL (example: `http://localhost:8000`)
- `VITE_FIREBASE_API_KEY`
- `VITE_FIREBASE_AUTH_DOMAIN`
- `VITE_FIREBASE_PROJECT_ID`
- `VITE_FIREBASE_STORAGE_BUCKET` (optional but recommended)
- `VITE_FIREBASE_MESSAGING_SENDER_ID` (optional but recommended)
- `VITE_FIREBASE_APP_ID`

## 2) Run

```bash
npm run dev
```

Build for production:

```bash
npm run build
npm run preview
```

## Notes

- Login uses Firebase email/password auth.
- Firebase ID token is resolved from Firebase Auth at request time and attached as:
  - `Authorization: Bearer <token>`
- Implemented routes:
  - `/login`
  - `/dashboard`
  - `/stores`
  - `/upload`
  - `/store/:storeId`
- For Firebase Hosting SPA deploys, `frontend/firebase.json` includes a catch-all rewrite to `/index.html` so deep links do not 404.
  - Deploy from the `frontend` directory with: `firebase deploy --only hosting --config firebase.json`
- Forecast preview tables intentionally show values only where backend horizons exist (`h=1/7/14`) and render `-` for the rest with a "More horizons coming" note.
