"""
Valida a conexao com AWS (RDS + S3) ANTES de rodar o dashboard.

Roda fora do Streamlit, entao os erros aparecem direto no terminal, com
mensagem clara -- e bem mais facil diagnosticar aqui do que dentro do app.

Uso:
    python test_aws_connection.py

Le as credenciais de (nesta ordem):
    1. .streamlit/secrets.toml  (mesmo arquivo que o app.py usa)
    2. variaveis de ambiente

Chaves esperadas:
    DATABASE_URL          postgresql://user:senha@endpoint:5432/antsactivematter2026
    S3_BUCKET             ants-active-matter-2026
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
"""

import os
import sys
from pathlib import Path

TABELAS_ESPERADAS = ["simulations", "phase_transition", "ant_trajectories"]

MODEL_FILES = [
    "model_1b_sigma_classifier.pt", "scaler_X_1b.pkl",
    "model_2_phi_predictor.pt", "scaler_X_2.pkl", "scaler_y_2.pkl",
    "model_3b_displacement.pt", "scaler_X_3b.pkl", "scaler_y_3b.pkl",
    "models_metadata.json",
]


def carregar_secrets():
    """Le .streamlit/secrets.toml (se existir) e cai para variaveis de ambiente."""
    secrets = {}
    caminho = Path(".streamlit/secrets.toml")

    if caminho.exists():
        try:
            try:
                import tomllib  # Python 3.11+
                with open(caminho, "rb") as f:
                    secrets = tomllib.load(f)
            except ImportError:
                import toml  # Python <3.11: pip install toml
                secrets = toml.load(caminho)
            print(f"[i] Credenciais lidas de {caminho}")
        except Exception as e:
            print(f"[!] Nao foi possivel ler {caminho}: {e}")
    else:
        print(f"[i] {caminho} nao encontrado -- usando variaveis de ambiente")

    for chave in ["DATABASE_URL", "S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]:
        if chave not in secrets and os.getenv(chave):
            secrets[chave] = os.getenv(chave)

    return secrets


def mascarar(url):
    """Esconde a senha ao imprimir a URL de conexao."""
    if not url or "://" not in url:
        return "(nao configurada)"
    esquema, resto = url.split("://", 1)
    if "@" in resto:
        credenciais, host = resto.split("@", 1)
        usuario = credenciais.split(":")[0]
        return f"{esquema}://{usuario}:****@{host}"
    return url


def testar_rds(db_url):
    print()
    print("=" * 66)
    print("1. RDS PostgreSQL")
    print("=" * 66)

    if not db_url:
        print("[--] DATABASE_URL nao configurada -- o app rodaria em modo local (CSV)")
        return False

    print(f"     URL: {mascarar(db_url)}")

    try:
        from sqlalchemy import create_engine, text
    except ImportError:
        print("[XX] sqlalchemy nao instalado -- rode: pip install -r requirements.txt")
        return False

    try:
        engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 15})
        with engine.connect() as conn:
            versao = conn.execute(text("SELECT version()")).fetchone()[0]
            print(f"[OK] Conectado: {versao.split(',')[0]}")

            # Tabelas presentes
            linhas = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' ORDER BY table_name
            """)).fetchall()
            existentes = {r[0] for r in linhas}

            tudo_ok = True
            for tabela in TABELAS_ESPERADAS:
                if tabela in existentes:
                    n = conn.execute(text(f"SELECT COUNT(*) FROM {tabela}")).fetchone()[0]
                    print(f"[OK] Tabela '{tabela}': {n:,} registros")
                    if n == 0:
                        print("     [!] vazia -- rode o ETL do notebook aws_setup_fixed.ipynb")
                        tudo_ok = False
                else:
                    print(f"[XX] Tabela '{tabela}' NAO existe")
                    tudo_ok = False

            # As queries que as abas novas usam sao as mais fraceis -- testar de fato
            if "ant_trajectories" in existentes:
                print()
                print("     Testando as queries das abas 'Formiga Individual' / 'Validacao':")

                conn.execute(text("""
                    SELECT ROUND((find / nav)::numeric, 2) as find_normalized,
                           AVG(CASE WHEN sigma = 1 THEN 1.0 ELSE 0.0 END) as p_puller,
                           COUNT(*) as n
                    FROM ant_trajectories GROUP BY find_normalized ORDER BY find_normalized
                """)).fetchall()
                print("     [OK] P(puller) vs find normalizado")

                conn.execute(text("""
                    SELECT ROUND(theta::numeric, 1) as theta_bin, AVG(phi), COUNT(*)
                    FROM ant_trajectories WHERE sigma = 1 GROUP BY theta_bin ORDER BY theta_bin
                """)).fetchall()
                print("     [OK] phi vs theta")

                import time
                t0 = time.time()
                conn.execute(text("""
                    WITH pairs AS (
                        SELECT theta, phi, sigma, nav, find, b, x, y,
                               LEAD(x) OVER (PARTITION BY run_id, site_id ORDER BY t) as x_next,
                               LEAD(y) OVER (PARTITION BY run_id, site_id ORDER BY t) as y_next
                        FROM ant_trajectories
                    )
                    SELECT theta, phi, sigma, nav, find, b,
                           (x_next - x) as dx, (y_next - y) as dy
                    FROM pairs WHERE x_next IS NOT NULL ORDER BY random() LIMIT 3000
                """)).fetchall()
                dt = time.time() - t0
                print(f"     [OK] amostra de deslocamento (LEAD + random) -- {dt:.1f}s")
                if dt > 20:
                    print("     [!] lenta na t3.micro; o cache do Streamlit (ttl=600) segura,")
                    print("         mas o primeiro carregamento da aba vai demorar isso")

            return tudo_ok

    except Exception as e:
        print(f"[XX] Falha ao conectar: {e}")
        print()
        print("     Checklist:")
        print("     - Instancia RDS com status 'available' no console AWS?")
        print("     - Security group libera a porta 5432 para o seu IP?")
        print("     - Endpoint, usuario, senha e nome do banco corretos na URL?")
        print("     - Nome do banco sem hifen (antsactivematter2026)?")
        return False


def testar_s3(bucket, chave, segredo):
    print()
    print("=" * 66)
    print("2. S3 (modelos de rede neural)")
    print("=" * 66)

    if not (bucket and chave and segredo):
        faltando = [n for n, v in [
            ("S3_BUCKET", bucket),
            ("AWS_ACCESS_KEY_ID", chave),
            ("AWS_SECRET_ACCESS_KEY", segredo),
        ] if not v]
        print(f"[--] Faltando: {', '.join(faltando)}")
        print("     As abas 'Formiga Individual' e 'Validacao do Modelo' ficariam")
        print("     desabilitadas (o resto do dashboard funciona normal).")
        return False

    print(f"     Bucket: {bucket}")

    try:
        import boto3
    except ImportError:
        print("[XX] boto3 nao instalado -- rode: pip install -r requirements.txt")
        return False

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=chave,
            aws_secret_access_key=segredo,
            region_name="us-east-1",
        )
        s3.head_bucket(Bucket=bucket)
        print("[OK] Bucket acessivel")

        resposta = s3.list_objects_v2(Bucket=bucket, Prefix="models/")
        presentes = {
            obj["Key"].split("/")[-1]: obj["Size"]
            for obj in resposta.get("Contents", [])
        }

        tudo_ok = True
        for nome in MODEL_FILES:
            if nome in presentes:
                print(f"[OK] models/{nome:<35} ({presentes[nome] / 1024:>8.1f} KB)")
            else:
                print(f"[XX] models/{nome:<35} AUSENTE")
                tudo_ok = False

        if not tudo_ok:
            print()
            print("     Rode o notebook upload_models_s3.ipynb para subir os que faltam.")

        return tudo_ok

    except Exception as e:
        print(f"[XX] Falha no S3: {e}")
        print()
        print("     Checklist:")
        print("     - Access key ainda ativa no IAM?")
        print("     - Nome do bucket correto e na regiao us-east-1?")
        print("     - A key tem permissao de s3:ListBucket e s3:GetObject nesse bucket?")
        return False


def main():
    print()
    print("=" * 66)
    print("VALIDACAO DE CONEXAO AWS -- Dashboard de Formigas")
    print("=" * 66)

    secrets = carregar_secrets()

    rds_ok = testar_rds(secrets.get("DATABASE_URL"))
    s3_ok = testar_s3(
        secrets.get("S3_BUCKET"),
        secrets.get("AWS_ACCESS_KEY_ID"),
        secrets.get("AWS_SECRET_ACCESS_KEY"),
    )

    print()
    print("=" * 66)
    print("RESUMO")
    print("=" * 66)
    print(f"  RDS (dados):            {'OK' if rds_ok else 'PENDENTE'}")
    print(f"  S3  (modelos):          {'OK' if s3_ok else 'PENDENTE'}")
    print()

    if rds_ok and s3_ok:
        print("  Tudo pronto. Rode:  streamlit run app.py")
        print("  As 6 abas devem funcionar com dados da nuvem.")
    elif rds_ok:
        print("  Dashboard de dados funciona (4 abas).")
        print("  As 2 abas de rede neural precisam do S3 -- veja acima.")
    else:
        print("  Corrija os itens marcados com [XX] antes de rodar o dashboard.")
        print("  (Sem DATABASE_URL o app cai em modo local, lendo CSVs de data/.)")
    print()

    return 0 if (rds_ok and s3_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
