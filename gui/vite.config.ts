import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

// Read allowed hosts from environment (comma-separated list) so we can configure the domain via compose
// Example: ALLOWED_HOSTS="pdp.s4home.dpdns.org,localhost"
const allowedHostsEnv = process.env.ALLOWED_HOSTS || 'pdp.shrishesha.space,localhost,0.0.0.0,::1,127.0.0.1';
const parseAllowedHosts = (v: string) => {
	if (!v) return ['localhost'];
	if (v === 'all' || v === '*') return 'all' as any;
	return v.split(',').map(s => s.trim()).filter(Boolean);
};
const allowedHosts = parseAllowedHosts(allowedHostsEnv);

// Allow the HOST env var to control interface binding; default to listening on all interfaces
const hostBinding: boolean | string = process.env.HOST ? process.env.HOST : true;

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
	},
	preview: {
		host: hostBinding,
		allowedHosts: allowedHosts as any,
	}
});
