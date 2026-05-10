"""
Cartola FC — Verificador de Rodada & Avaliador do Jogador
=========================================================
Regras heurísticas para decidir, com poucos dados, quais médias são confiáveis:

  Verificador de Rodada
    rodada <= 4   → "insufficient":  nenhuma média é confiável; cair em features básicas
    rodada 5–10   → "general_only":  use a média geral, mas evite a de mando de campo
    rodada >= 11  → "full":          zona segura, prossiga para avaliar o jogador

  Avaliador do Jogador (gap relativo de 30% por padrão)
    desempenho similar    → usa média geral
    grande diferença      → calcula a diferença e devolve a média do mando atual
    sem dados / unknown   → devolve média geral como fallback seguro
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional

GAP_RELATIVO_PADRAO = 0.30


@dataclass
class RegimeRodada:
    rodada: int
    regime: str                      # "insufficient" | "general_only" | "full"
    use_general_mean: bool
    use_home_away: bool
    message: str


@dataclass
class AvaliacaoJogador:
    decision: str                    # "similar" | "big_diff" | "unknown"
    media_recomendada: Optional[float]
    gap_absoluto: Optional[float]
    gap_relativo: Optional[float]
    explicacao: str


def verificar_rodada(rodada: int) -> RegimeRodada:
    """Aplica a tabela de regimes do Verificador de Rodada."""
    if rodada is None or rodada <= 4:
        return RegimeRodada(
            rodada=rodada,
            regime="insufficient",
            use_general_mean=False,
            use_home_away=False,
            message=f"R{rodada}: nenhuma média é confiável ainda.",
        )
    if rodada <= 10:
        return RegimeRodada(
            rodada=rodada,
            regime="general_only",
            use_general_mean=True,
            use_home_away=False,
            message=f"R{rodada}: use média geral, evite a de mando de campo.",
        )
    return RegimeRodada(
        rodada=rodada,
        regime="full",
        use_general_mean=True,
        use_home_away=True,
        message=f"R{rodada}: zona segura, prossiga para avaliar o jogador.",
    )


def _is_num(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def avaliar_jogador(
    media_geral: Optional[float],
    media_casa: Optional[float] = None,
    media_fora: Optional[float] = None,
    mando: int = 0,
    threshold: float = GAP_RELATIVO_PADRAO,
) -> AvaliacaoJogador:
    """
    Decide qual média usar para o jogador no contexto do mando atual.

    `mando`: 1 (jogo em casa), -1 (jogo fora), 0 (desconhecido — força fallback).
    Quando casa/fora ausentes ou mando=0, devolve "unknown" e recomenda a geral.
    Quando o gap relativo |casa - fora| / max(casa, fora) >= threshold (e ambos
    >0), devolve "big_diff" com a média do mando atual. Caso contrário "similar"
    (usa a geral).
    """
    if not _is_num(media_casa) or not _is_num(media_fora) or mando == 0:
        return AvaliacaoJogador(
            decision="unknown",
            media_recomendada=media_geral if _is_num(media_geral) else None,
            gap_absoluto=None,
            gap_relativo=None,
            explicacao="Dados de casa/fora indisponíveis ou mando desconhecido — fallback média geral.",
        )

    abs_gap = abs(media_casa - media_fora)
    base = max(abs(media_casa), abs(media_fora), 1e-9)
    rel_gap = abs_gap / base

    if rel_gap >= threshold:
        media_mando = media_casa if mando == 1 else media_fora
        return AvaliacaoJogador(
            decision="big_diff",
            media_recomendada=media_mando,
            gap_absoluto=abs_gap,
            gap_relativo=rel_gap,
            explicacao=(
                f"Gap {rel_gap:.0%} >= {threshold:.0%}: usar média de "
                f"{'casa' if mando == 1 else 'fora'} ({media_mando:.2f})."
            ),
        )

    return AvaliacaoJogador(
        decision="similar",
        media_recomendada=media_geral if _is_num(media_geral) else (media_casa + media_fora) / 2,
        gap_absoluto=abs_gap,
        gap_relativo=rel_gap,
        explicacao=f"Gap {rel_gap:.0%} < {threshold:.0%}: desempenho similar, usar média geral.",
    )


def _interativo() -> None:
    """CLI: roda o verificador de rodada e, se aplicável, o avaliador de jogador."""
    try:
        rodada = int(input("Rodada alvo: ").strip())
    except ValueError:
        print("Rodada inválida.")
        return

    regime = verificar_rodada(rodada)
    print(f"\n[Verificador] {regime.message}  (regime={regime.regime})")

    if regime.regime == "insufficient":
        print("Sem dados suficientes — nenhuma média é confiável. Use preço/posição apenas.")
        return

    if regime.regime == "general_only":
        try:
            mg = float(input("Média geral do jogador: ").strip())
        except ValueError:
            print("Média inválida.")
            return
        print(f"Recomendado: usar média geral = {mg:.2f} (mando ignorado).")
        return

    print("\n[Avaliador] Escolha uma opção:")
    print("  1) Desempenho similar (vai direto para média geral)")
    print("  2) Grande diferença (vou digitar média de casa e fora)")
    print("  3) Não sei ainda (recomenda média geral como padrão)")
    op = input("Opção [1/2/3]: ").strip()

    try:
        mg = float(input("Média geral: ").strip())
    except ValueError:
        print("Média inválida.")
        return

    if op == "1":
        av = avaliar_jogador(mg, media_casa=mg, media_fora=mg, mando=0)
        print(f"→ {av.decision}: {av.explicacao}")
        print(f"   Média recomendada: {av.media_recomendada:.2f}")
        return

    if op == "2":
        try:
            mc = float(input("Média em casa: ").strip())
            mf = float(input("Média fora:    ").strip())
            mando = int(input("Mando atual (1=casa, -1=fora): ").strip())
        except ValueError:
            print("Entrada inválida.")
            return
        av = avaliar_jogador(mg, media_casa=mc, media_fora=mf, mando=mando)
        print(f"→ {av.decision}: {av.explicacao}")
        if av.media_recomendada is not None:
            print(f"   Média recomendada: {av.media_recomendada:.2f}")
        return

    av = avaliar_jogador(mg, media_casa=None, media_fora=None, mando=0)
    print(f"→ {av.decision}: {av.explicacao}")
    if av.media_recomendada is not None:
        print(f"   Média recomendada: {av.media_recomendada:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartola FC — Verificador de Rodada / Avaliador do Jogador")
    parser.add_argument("--rodada", type=int, default=None, help="Rodada alvo (modo não-interativo)")
    args = parser.parse_args()

    if args.rodada is not None:
        regime = verificar_rodada(args.rodada)
        print(f"R{regime.rodada}: regime={regime.regime} | "
              f"general={regime.use_general_mean} | home_away={regime.use_home_away}")
        print(f"  → {regime.message}")
    else:
        _interativo()
