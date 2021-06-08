export function argmax(array: number[]): number{
    return [].reduce.call(array, (_m: unknown, _c: never, _i: number, _arr: never[]) => _c > _arr[_m as number] ? _i : _m, 0) as number;
}
