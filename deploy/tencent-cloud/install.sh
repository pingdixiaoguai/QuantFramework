#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="/opt/QuantFramework"
service_user="quant"
service_group="quant"
service_home="/var/lib/quantframework"
uv_cache_dir="${service_home}/.cache/uv"
uv_python_dir="${service_home}/python"
env_dir="/etc/quantframework"
env_file="${env_dir}/quant.env"
uv_version="${UV_VERSION:-0.11.32}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this installer as root: sudo bash deploy/tencent-cloud/install.sh" >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd -- "${script_dir}/../.." && pwd)"

if [[ "${source_root}" != "${project_dir}" ]]; then
    echo "Project must be deployed at ${project_dir}." >&2
    echo "Current checkout: ${source_root}" >&2
    echo "Move or clone the repository to ${project_dir}, then rerun this script." >&2
    exit 1
fi

if ! id -u "${service_user}" >/dev/null 2>&1; then
    useradd \
        --system \
        --user-group \
        --home-dir "${service_home}" \
        --create-home \
        --shell /usr/sbin/nologin \
        "${service_user}"
fi

install -d -m 0750 -o "${service_user}" -g "${service_group}" "${service_home}"
install -d -m 0750 -o "${service_user}" -g "${service_group}" \
    "${uv_cache_dir}" \
    "${uv_python_dir}"
install -d -m 0750 -o root -g "${service_group}" "${env_dir}"
install -d -m 0750 -o "${service_user}" -g "${service_group}" \
    "${project_dir}/data/db" \
    "${project_dir}/state"
chown -R "${service_user}:${service_group}" "${project_dir}"

if [[ ! -f "${env_file}" ]]; then
    install \
        -m 0640 \
        -o root \
        -g "${service_group}" \
        "${script_dir}/quant.env.example" \
        "${env_file}"
    echo "Created ${env_file}." >&2
    echo "Edit its Tushare and DingTalk values, then rerun this installer." >&2
    exit 2
fi

for required_key in TUSHARE_TOKEN DINGTALK_WEBHOOK; do
    if ! grep -Eq "^${required_key}=(\"[^\"]+\"|[^[:space:]]+)$" "${env_file}"; then
        echo "${env_file} has no usable ${required_key} value." >&2
        exit 2
    fi
done

if grep -Eq "your_(tushare_token|access_token)" "${env_file}"; then
    echo "${env_file} still contains placeholder credentials." >&2
    exit 2
fi

if [[ ! -x /usr/bin/flock ]]; then
    echo "/usr/bin/flock is required (normally provided by util-linux)." >&2
    exit 1
fi

if [[ ! -x /usr/local/bin/uv ]]; then
    if ! command -v curl >/dev/null 2>&1; then
        echo "curl is required to install uv." >&2
        exit 1
    fi
    echo "Installing uv ${uv_version} to /usr/local/bin ..."
    curl -LsSf "https://astral.sh/uv/${uv_version}/install.sh" \
        | env UV_UNMANAGED_INSTALL=/usr/local/bin sh
fi

echo "Synchronizing the locked Python 3.12 production environment ..."
(
    cd "${project_dir}"
    runuser -u "${service_user}" -- \
        /usr/bin/env \
        UV_CACHE_DIR="${uv_cache_dir}" \
        UV_PYTHON_INSTALL_DIR="${uv_python_dir}" \
        /usr/local/bin/uv sync --frozen --no-dev
    runuser -u "${service_user}" -- \
        "${project_dir}/.venv/bin/python" \
        -m scripts.run_daily_job \
        --help \
        >/dev/null
)

install -m 0644 \
    "${script_dir}/quant-daily.service" \
    /etc/systemd/system/quant-daily.service
install -m 0644 \
    "${script_dir}/quant-daily.timer" \
    /etc/systemd/system/quant-daily.timer

timedatectl set-timezone Asia/Shanghai
systemctl daemon-reload
systemd-analyze verify \
    /etc/systemd/system/quant-daily.service \
    /etc/systemd/system/quant-daily.timer
systemctl enable --now quant-daily.timer

echo
echo "QuantFramework timer installed."
echo "Next run:"
systemctl list-timers quant-daily.timer --no-pager
echo
echo "Run one manual verification when ready:"
echo "  systemctl start quant-daily.service"
echo "  journalctl -u quant-daily.service -n 200 --no-pager"
