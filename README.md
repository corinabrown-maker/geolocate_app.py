README.md (Quick Deploy)

1) Create a repo

Make a new GitHub repo (public or private) and add the three files above.

2) Deploy to Streamlit Community Cloud (free)

Go to https://streamlit.io/cloud and sign in with GitHub.

New app → Select your repo.

Main file path: streamlit_app.py.

Deploy. First build may take a few minutes.

After deploy, you’ll get a URL like https://<your-app>.streamlit.app that you can open from any Intune-managed computer in a browser—no installs required.

3) Usage

Convert DWG → DXF if needed (ODA File Converter).

Open your Streamlit URL, upload DXF, enter control points, pick CRS (e.g., 2193), transform, and download outputs.

4) Notes & Safety

Uploaded files are processed in memory on the app server. Avoid uploading sensitive CAD if the app is public; set repo to private and restrict access if needed.

For New Zealand projects, EPSG:2193 (NZTM2000, metres) is recommended.

With ≥3 control points the app reports RMSE to assess fit quality.

5) Optional: Private/Enterprise hosting

You can host this on Azure App Service or Streamlit on Azure behind AAD logins for internal-only access.
