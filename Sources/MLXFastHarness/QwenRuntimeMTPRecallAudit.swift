import Foundation
import MLX
import MLXFastModel

/// E155 draft-head recall audit. UNCOMMITTED RESEARCH PATCH — never submitted.
///
/// For every proposal slot of a decode run this records four token ids that
/// share one proposal hidden row:
///
/// * `ann`            — the id the shipped ANN + affine-4 rerank path proposed.
/// * `exact_compact`  — the exact argmax over the 98,330 real compact rows.
/// * `exact_full`     — the exact argmax over all 248,320 lm_head rows.
/// * `target`         — the target verify row argmax, i.e. the truth for the slot.
///
/// The compact rows are a gather of the backbone `lm_head` rows, so both exact
/// argmaxes come from one `applyLMHead` call and `exact_compact` is a strict
/// restriction of `exact_full`. Nothing here runs unless
/// `MLX_E155_RECALL_AUDIT=1` is set, and nothing here feeds the session back.
enum QwenMTPRecallAudit {
    static let prefixCount = 98_304
    static let controlStart = 248_044
    static let controlEnd = 248_070
    static let realCount = 98_330

    private nonisolated(unsafe) static var sink: FileHandle?
    private nonisolated(unsafe) static var positions: MLXArray?

    static func installIfRequested(model: any Qwen36MTPTarget) {
        let env = ProcessInfo.processInfo.environment
        guard env["MLX_E155_RECALL_AUDIT"] == "1" else {
            fputs("mlxfast-worker: e155_recall_audit=off\n", stderr)
            return
        }
        guard let path = env["MLX_E155_RECALL_AUDIT_PATH"], !path.isEmpty else {
            fputs(
                "mlxfast-worker: e155_recall_audit=on_but_no_path\n", stderr)
            return
        }
        FileManager.default.createFile(atPath: path, contents: nil)
        guard let handle = FileHandle(forWritingAtPath: path) else {
            fputs("mlxfast-worker: e155_recall_audit=path_open_failed\n", stderr)
            return
        }
        sink = handle
        positions = arange(realCount, dtype: .int32)
        fputs("mlxfast-worker: e155_recall_audit=on path=\(path)\n", stderr)

        Qwen36MTPBlockSession.draftAuditHook = {
            round, hiddens, drafts, verifyArgmax, accepted in
            record(
                model: model, round: round, hiddens: hiddens, drafts: drafts,
                verifyArgmax: verifyArgmax, accepted: accepted)
        }
    }

    private static func record(
        model: any Qwen36MTPTarget, round: Int, hiddens: [MLXArray],
        drafts: [Int], verifyArgmax: [Int], accepted: Int
    ) {
        guard let sink, let positions, !hiddens.isEmpty else { return }
        let slots = min(hiddens.count, min(drafts.count, verifyArgmax.count))
        var ids: [MLXArray] = []
        var values: [MLXArray] = []
        for slot in 0 ..< slots {
            let flat = model.applyLMHead(hiddens[slot])
                .reshaped([-1]).asType(.float32)
            let compact = concatenated(
                [flat[0 ..< prefixCount], flat[controlStart ..< controlEnd]],
                axis: 0)
            let compactIdx = argMax(compact, axis: 0).asType(.int32)
            // Positive control: remove the exact winner and re-reduce. A real
            // reduction must then name a different row; a tautological one
            // cannot fail. The masked maximum is also the runner-up value, so
            // the same mask measures how close the decision was.
            let masked = which(
                positions .!= compactIdx, compact, MLXArray(Float(-3.0e38)))
            ids.append(stacked([
                argMax(flat, axis: 0).asType(.int32),
                compactIdx,
                argMax(masked, axis: 0).asType(.int32),
            ]))
            values.append(stacked([
                flat[drafts[slot]],
                flat[verifyArgmax[slot]],
                flat.max(),
                compact.max(),
                masked.max(),
            ]))
        }
        let idBlock = stacked(ids)
        let valueBlock = stacked(values)
        eval(idBlock, valueBlock)
        let flatIDs = idBlock.asArray(Int32.self).map { Int($0) }
        let flatValues = valueBlock.asArray(Float.self)

        var text = ""
        for slot in 0 ..< slots {
            let compactRaw = flatIDs[slot * 3 + 1]
            let maskedRaw = flatIDs[slot * 3 + 2]
            text += "{\"round\":\(round),\"slot\":\(slot)"
                + ",\"d\":\(drafts.count),\"accepted\":\(accepted)"
                + ",\"ann\":\(drafts[slot])"
                + ",\"exact_compact\":\(mapCompact(compactRaw))"
                + ",\"exact_full\":\(flatIDs[slot * 3])"
                + ",\"masked_compact\":\(mapCompact(maskedRaw))"
                + ",\"target\":\(verifyArgmax[slot])"
                + ",\"ann_logit\":\(flatValues[slot * 5])"
                + ",\"target_logit\":\(flatValues[slot * 5 + 1])"
                + ",\"full_max\":\(flatValues[slot * 5 + 2])"
                + ",\"compact_max\":\(flatValues[slot * 5 + 3])"
                + ",\"compact_second\":\(flatValues[slot * 5 + 4])}\n"
        }
        sink.write(Data(text.utf8))
    }

    /// Compact row index back to the target tokenizer id, matching the gather
    /// that `makeCompactDraftHead` performs.
    private static func mapCompact(_ index: Int) -> Int {
        index < prefixCount ? index : controlStart + (index - prefixCount)
    }
}
